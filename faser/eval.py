import json
import logging
import os
import pickle
import random
import time
from statistics import mean, median
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
from tokenizers import Tokenizer
from torch import cosine_similarity
from tqdm import tqdm
from transformers import LongformerModel

from faser.dataset import Bin2MLFunctionString
from faser.default_config import return_default_longformer_config
from faser.metrics import eval_model_no_model
from faser.nets.wrappers import MetricLearnHFWrapper


class FaserGeneralFuncSearchEval:
    def __init__(
        self,
        torch_model_bin_fp,
        eval_data_fp: str,
        max_seq_len: int,
        tokeniser_fp: str,
        filter_str: Tuple[None, str],
        num_eval_search_pools: int,
        search_pool_size: int,
        model_out_size: int = 768,
        embed_out_size: int = 128,
    ):
        self.torch_model_bin = torch_model_bin_fp
        self.eval_data_fp = eval_data_fp
        self.max_seq_len = max_seq_len
        self.tokeniser_fp = tokeniser_fp
        self.filter_str = filter_str
        self.num_eval_search_pools = num_eval_search_pools
        self.search_pool_size = search_pool_size
        self.model_out_size = model_out_size
        self.embed_out_size = embed_out_size
        self.vocab_size = Tokenizer.from_file(self.tokeniser_fp).get_vocab_size()
        self.model_config = return_default_longformer_config(self.vocab_size)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def load_eval_dataset(self):
        return Bin2MLFunctionString(
            self.eval_data_fp,
            max_seq_len=self.max_seq_len,
            tokeniser_fp=self.tokeniser_fp,
            dynamic_encoding=True,
            remove_uniques=True,
            with_labels=True,
            filter_str=self.filter_str,
        )

    def embed_function(self, model: nn.Module, data: Tuple[Dict, Dict]):
        data = {
            key: torch.unsqueeze(torch.LongTensor(value), dim=0)
            for key, value in data.items()
        }
        data = {key: value.to(self.device) for key, value in data.items()}
        embeddings = model.forward(
            data["ids"], data["attention_mask"], data["type_ids"]
        )

        return (embeddings, data["label"])

    def load_model(self):
        encoder = LongformerModel(self.model_config, add_pooling_layer=True)
        model = MetricLearnHFWrapper(
            encoder,
            model_out_size=self.model_out_size,
            embed_out_size=self.embed_out_size,
        )
        if self.device == "cpu":
            model.load_state_dict(
                torch.load(self.torch_model_bin, map_location=torch.device("cpu"))
            )
        else:
            model.load_state_dict(torch.load(self.torch_model_bin))

        model.to(self.device)
        model.eval()
        return model

    def eval(self):
        dataset = self.load_eval_dataset()
        model = self.load_model()

        logging.info("Starting sampling...")
        start_time = time.time()
        num_funcs = len(dataset.function_objs)
        # Sample positives
        num_pos = self.num_eval_search_pools
        num_neg = self.search_pool_size

        pos_pair_labels = random.choices(dataset.labels, k=num_pos)

        scores = []
        target = []  # True = Are Sim False = Not Sim
        indexes = []

        logging.info("Creating Search Pools...")
        for i, label in tqdm(enumerate(pos_pair_labels), total=num_pos):
            pos_pair = []
            neg_pairs = []
            offset_samples = random.sample(range(dataset.label_counter[label]), k=2)
            assert (
                dataset.function_objs[dataset.labels.index(label) + offset_samples[0]][
                    "label"
                ]
                == dataset.function_objs[
                    dataset.labels.index(label) + offset_samples[1]
                ]["label"]
            )
            pos_pair.append(
                (
                    dataset[dataset.labels.index(label) + offset_samples[0]],
                    dataset[dataset.labels.index(label) + offset_samples[1]],
                )
            )

            while len(neg_pairs) != num_neg:
                random_index = random.sample(range(num_funcs), k=1)[0]
                if (
                    dataset.function_objs[
                        dataset.labels.index(label) + offset_samples[0]
                    ]["label"]
                    != dataset.function_objs[random_index]["label"]
                ):
                    neg_pairs.append(
                        (
                            dataset[dataset.labels.index(label) + offset_samples[0]],
                            dataset[random_index],
                        )
                    )
                else:
                    continue

            logging.debug("Embedding sampled Pairs")
            for a, b in pos_pair + neg_pairs:
                with torch.no_grad():
                    logging.debug("Embedding A...")
                    a, al = self.embed_function(model, a)
                    logging.debug("Embedding B...")
                    b, bl = self.embed_function(model, b)

                    logging.debug("Calculating Sim...")
                    sim = torch.cosine_similarity(a, b, dim=1).item()
                    scores.append(sim)
                    target.append(True if bl == label else False)
                    indexes.append(i)

        logging.info("Calculating metrics!")
        indexes, scores, target = (
            torch.LongTensor(indexes),
            torch.FloatTensor(scores),
            torch.LongTensor(target),
        )

        metric_dict = eval_model_no_model(indexes, scores, target)
        print(metric_dict)

        print("Total Time: %s seconds\n" % (time.time() - start_time))


class FaserVulnSearchEval:
    NETGEAR_VULNS = ["CMS_decrypt", "PKCS7_dataDecode", "MDC2_Update", "BN_bn2dec"]
    TPLINK_VULNS = [
        "CMS_decrypt",
        "PKCS7_dataDecode",
        "BN_bn2dec",
        "X509_NAME_oneline",
        "EVP_EncryptUpdate",
        "EVP_EncodeUpdate",
        "SRP_VBASE_get_by_user",
        "BN_dec2bn",
        "BN_hex2bn",
    ]

    def __init__(
        self,
        torch_model_bin_fp: str,
        eval_data_dir: str,
        max_seq_len: int,
        tokeniser_fp: str,
        model_friendly_name: str,
        model_out_size: int = 768,
        embed_out_size: int = 128,
    ):
        self.torch_model_bin = torch_model_bin_fp
        self.eval_data_dir = eval_data_dir
        self.max_seq_len = max_seq_len
        self.tokeniser_fp = tokeniser_fp
        self.model_out_size = model_out_size
        self.embed_out_size = embed_out_size
        self.vocab_size = Tokenizer.from_file(self.tokeniser_fp).get_vocab_size()
        self.model_config = return_default_longformer_config(self.vocab_size)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = self.load_model()
        self.tokeniser = Tokenizer.from_file(self.tokeniser_fp)

        # Processed function strings from the firmware libcrypto versions
        self.netgear_path = f"{self.eval_data_dir}/libcrypto.so.1.0.0_NETGEAR_R7000_1.0.2h_arm32-efs.json"
        self.tplink_path = f"{self.eval_data_dir}/libcrypto.so.1.0.0_TP-Link_Deco-M4_1.0.2d_mips32-efs.json"

        # Processed function strings from the compiled libcrypto versions
        self.mips_targets = (
            f"{self.eval_data_dir}/libcrypto.so.1.0.0_openssl_1.0.2d_mips32-efs.json"
        )
        self.arm_targets = (
            f"{self.eval_data_dir}/libcrypto.so.1.0.0_openssl_1.0.2d_arm32-efs.json"
        )
        self.x64_targets = (
            f"{self.eval_data_dir}/libcrypto.so.1.0.0_openssl_1.0.2d_x64-efs.json"
        )
        self.x86_targets = (
            f"{self.eval_data_dir}/libcrypto.so.1.0.0_openssl_1.0.2d_x86-efs.json"
        )
        self.riscv_targets = (
            f"{self.eval_data_dir}/libcrypto.so.1.0.0_openssl_1.0.2d_riscv32-efs.json"
        )

        self.model_friendly_name = model_friendly_name

    def load_model(self):
        encoder = LongformerModel(self.model_config, add_pooling_layer=True)
        model = MetricLearnHFWrapper(
            encoder,
            model_out_size=self.model_out_size,
            embed_out_size=self.embed_out_size,
        )
        if self.device == "cpu":
            model.load_state_dict(
                torch.load(self.torch_model_bin, map_location=torch.device("cpu"))
            )
        else:
            model.load_state_dict(torch.load(self.torch_model_bin))

        model.to(self.device)
        model.eval()

        return model

    def single_architecture_rank(
        self,
        firmware_name: str,
        firmware_funcs: str,
        arch_query_funcs: str,
        firmware_vulns: List[str],
    ):
        sp_save_path = f"{self.model_friendly_name}-{firmware_name}-sp.pickle"

        sp = []
        # Load Searchpool data
        tp_link_sp = json.load(open(firmware_funcs))

        for k, v in tp_link_sp.items():
            sp.append((k, v))

        # Load target funcs
        targets = []

        all_targets = json.load(open(arch_query_funcs))
        for k, v in all_targets.items():
            if k[4:] in firmware_vulns:
                targets.append((k, v))

        # Generate Embeddings of search pool
        search_pool_embeddings = []
        if os.path.exists(sp_save_path):
            search_pool_embeddings = pickle.load(open(sp_save_path, "rb"))
        else:
            with torch.inference_mode():
                for k, v in tqdm(sp):
                    tokenised = self.tokeniser.encode(v)
                    output = {
                        "ids": tokenised.ids,
                        "attention_mask": tokenised.attention_mask,
                        "type_ids": tokenised.type_ids,
                    }
                    output = {
                        key: torch.unsqueeze(torch.LongTensor(value), dim=0)
                        for key, value in output.items()
                    }
                    output = {
                        key: value.to(self.device) for key, value in output.items()
                    }
                    embedded = self.model.forward(
                        output["ids"], output["attention_mask"], output["type_ids"]
                    )
                    search_pool_embeddings.append((k, embedded))
            with open(sp_save_path, "wb") as f:
                pickle.dump(search_pool_embeddings, f)

        # Generate Embeddings of targets
        target_embeddings = []
        with torch.inference_mode():
            for k, v in tqdm(targets):
                tokenised = self.tokeniser.encode(v)
                output = {
                    "ids": tokenised.ids,
                    "attention_mask": tokenised.attention_mask,
                    "type_ids": tokenised.type_ids,
                }
                output = {
                    key: torch.unsqueeze(torch.LongTensor(value), dim=0)
                    for key, value in output.items()
                }
                output = {key: value.to(self.device) for key, value in output.items()}
                embedded = self.model.forward(
                    output["ids"], output["attention_mask"], output["type_ids"]
                )
                target_embeddings.append((k, embedded))
        # Calculate Sims

        ranks = []
        sim_hits = []
        for name, target in target_embeddings:
            sims = []
            names = []

            for sp_name, sp_embed in search_pool_embeddings:
                sim = cosine_similarity(target, sp_embed)
                sims.append(sim)
                names.append(sp_name)

            zipped = list(zip(sims, names))
            zipped.sort(reverse=True)

            for i, z in enumerate(zipped):
                if name == z[1]:
                    print(
                        f"Found {name} at {i + 1} with a sim score of {round(z[0].item(), 4)} when compared {z[1]}"
                    )
                    ranks.append(i + 1)
                    sim_hits.append(round(z[0].item(), 4))
        print(
            f"\t{ranks} - Mean Rank: {round(mean(ranks))} Median Rank: {median(ranks)}"
        )
        print(f"\t{sim_hits}")

    def rank(self):
        print(f"Geneating results for Netgear vuln search ...")
        print("ARM -> ARM")
        self.single_architecture_rank(
            "netgear", self.netgear_path, self.arm_targets, self.NETGEAR_VULNS
        )
        print("MIPS -> ARM")
        self.single_architecture_rank(
            "netgear", self.netgear_path, self.mips_targets, self.NETGEAR_VULNS
        )
        print("X86 -> ARM")
        self.single_architecture_rank(
            "netgear", self.netgear_path, self.x86_targets, self.NETGEAR_VULNS
        )
        print("X64 -> ARM")
        self.single_architecture_rank(
            "netgear", self.netgear_path, self.x64_targets, self.NETGEAR_VULNS
        )
        print("riscv -> ARM")
        self.single_architecture_rank(
            "netgear", self.netgear_path, self.riscv_targets, self.NETGEAR_VULNS
        )

        print("Geneating results for TP-Link vuln search...")
        print("MIPS -> MIPS")
        self.single_architecture_rank(
            "tplink", self.tplink_path, self.mips_targets, self.TPLINK_VULNS
        )
        print("ARM -> MIPS")
        self.single_architecture_rank(
            "tplink", self.tplink_path, self.arm_targets, self.TPLINK_VULNS
        )
        print("X86 -> MIPS")
        self.single_architecture_rank(
            "tplink", self.tplink_path, self.x86_targets, self.TPLINK_VULNS
        )
        print("X64 -> MIPS")
        self.single_architecture_rank(
            "tplink", self.tplink_path, self.x64_targets, self.TPLINK_VULNS
        )
        print("riscv -> MIPS")
        self.single_architecture_rank(
            "tplink", self.tplink_path, self.riscv_targets, self.TPLINK_VULNS
        )
