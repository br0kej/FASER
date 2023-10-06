import torch.nn as nn


class MetricLearnHFWrapper(nn.Module):
    """
    A wrapper class that puts a small dense network after
    an encoder model (a huggingface model)
    """

    def __init__(
        self,
        encoder_model,
        model_out_size: int = 128,
        embed_out_size: int = 128,
        freeze_encoder: bool = False,
    ):
        super(MetricLearnHFWrapper, self).__init__()
        self.encoder = encoder_model
        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False

        self.dense_1 = nn.Linear(model_out_size, model_out_size // 2)
        self.dense_2 = nn.Linear(model_out_size // 2, embed_out_size)
        self.act = nn.ReLU()

    def forward(self, x, attention_mask, token_type_ids):
        # Encode
        x = self.encoder(
            x,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            output_hidden_states=True,
        )

        x = self.act(self.dense_1(x.pooler_output))

        x = self.act(self.dense_2(x))

        return x
