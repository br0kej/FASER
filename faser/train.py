"""
What do we need in here?

 - Generic Torch training loop with metric generation
 - Wandb support to store metrics
 - Model saving
"""
import itertools
from statistics import mean

import torch
from pytorch_metric_learning import losses, miners, samplers
from pytorch_metric_learning.distances import CosineSimilarity
from tokenizers import Tokenizer
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import LongformerModel

from faser.dataset import Bin2MLFunctionString
from faser.default_config import return_default_longformer_config
from faser.metrics import eval_model_no_model
from faser.nets.wrappers import MetricLearnHFWrapper


class FASERTrain:
    def __init__(
        self,
        name: str,
        train_data_fp: str,
        test_data_fp: str,
        tokeniser_fp: str,
        batch_size: int,
        num_pos_pairs_in_batch: int,
        learning_rate: float,
        num_training_epochs: int,
        num_accumlation_steps: int,
        filter_str=None,
    ):
        self.name = name
        self.train_data_fp = train_data_fp
        self.test_data_fp = test_data_fp
        self.tokeniser_fp = tokeniser_fp
        self.vocab_size = Tokenizer.from_file(self.tokeniser_fp).get_vocab_size()
        self.model_config = return_default_longformer_config(self.vocab_size)
        self.batch_size = batch_size
        self.num_pos_pairs_in_batch = num_pos_pairs_in_batch
        self.learning_rate = learning_rate
        self.num_training_epochs = num_training_epochs
        self.num_accumlation_steps = num_accumlation_steps

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.train_dataset = Bin2MLFunctionString(
            self.train_data_fp,
            max_seq_len=self.model_config.max_position_embeddings,
            tokeniser_fp=self.tokeniser_fp,
            dynamic_encoding=True,
            filter_str=filter_str,
        )
        self.test_dataset = Bin2MLFunctionString(
            self.test_data_fp,
            max_seq_len=self.model_config.max_position_embeddings,
            tokeniser_fp=self.tokeniser_fp,
            dynamic_encoding=True,
        )

        self.sampler = samplers.MPerClassSampler(
            self.train_dataset.labels,
            self.num_pos_pairs_in_batch,
            self.batch_size,
            length_before_new_iter=100000,
        )

        self.train_dl = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            sampler=self.sampler,
            pin_memory=True,
            num_workers=4,
        )

        self.encoder = LongformerModel(self.model_config, add_pooling_layer=True)
        self.model = MetricLearnHFWrapper(
            self.encoder, model_out_size=768, embed_out_size=128
        ).to(self.device)

        self.miner = miners.BatchHardMiner(distance=CosineSimilarity())
        self.loss_func = losses.CircleLoss(m=0.25, gamma=256)
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.learning_rate
        )

    def train(self):
        self.optimizer.zero_grad()

        # Training Loop
        avg_losses = []
        losses = []
        batch_counter = 0

        for epoch in range(self.num_training_epochs):
            for i, data in tqdm(
                enumerate(self.train_dl),
                total=len(self.train_dl),
                desc=f"Epoch: {epoch}",
            ):
                batch_counter += 1
                # Move data onto GPU
                data = {key: value.to(self.device) for key, value in data.items()}

                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    # Forward Pass of Model
                    embeddings = self.model(
                        data["ids"], data["attention_mask"], data["type_ids"]
                    )

                    # Mine embeddings
                    hard_pairs = self.miner(embeddings, data["label"])

                    # Calculate Loss
                    loss = self.loss_func(embeddings, data["label"], hard_pairs)

                    loss.backward()
                    losses.append(loss.item())
                if ((i + 1) % self.num_accumlation_steps == 0) or (
                    i + 1 == len(self.train_dl)
                ):
                    avg_batch_loss = mean(losses)
                    avg_losses.append(avg_batch_loss)
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    print(
                        f"Loss at {epoch}|{i + 1}: {mean(losses)} - {mean(avg_losses)}"
                    )
                    losses = []

                if i > 0:
                    if i % 2500 == 0:
                        torch.save(
                            self.model.state_dict(), f"{self.name}-{batch_counter - 1}"
                        )
                    if i % 500 == 0:
                        self.model.eval()
                        func_embeds = torch.Tensor()
                        func_labels = torch.Tensor()

                        for _, data in tqdm(
                            enumerate(DataLoader(self.test_dataset, batch_size=1)),
                            total=len(self.test_dataset),
                        ):
                            with torch.no_grad():
                                data = {
                                    key: value.to(self.device)
                                    for key, value in data.items()
                                }
                                embeddings = self.model.forward(
                                    data["ids"],
                                    data["attention_mask"],
                                    data["type_ids"],
                                )
                                func_embeds = torch.cat((func_embeds, embeddings.cpu()))
                                func_labels = torch.cat(
                                    (func_labels, data["label"].cpu())
                                )

                        scores = []
                        target = []  # True = Are Sim False = Not Sim
                        indexes = []
                        for a, b in itertools.combinations(
                            zip(func_embeds, func_labels), 2
                        ):
                            sim = torch.cosine_similarity(a[0], b[0], dim=0).item()
                            scores.append(sim)
                            target.append(True if a[1] == b[1] else False)
                            indexes.append(a[1].item())

                        metric_dict = eval_model_no_model(
                            torch.LongTensor(indexes),
                            torch.Tensor(scores),
                            torch.Tensor(target),
                        )
                        print(f"{epoch} - {metric_dict}")

                        self.model.train()
