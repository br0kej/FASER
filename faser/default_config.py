from transformers import LongformerConfig


def return_default_longformer_config(vocab_size: int):
    """
    The default longformer congif used for FASER

    Args:
        vocab_size: The size of the vocabulary used

    Returns:
        A LongFormerConfig() with the correct values
    """
    config = LongformerConfig()
    config.num_attention_heads = 8
    config.num_hidden_layers = 8
    config.vocab_size = vocab_size
    config.intermediate_size = 2048
    config.max_position_embeddings = 4096
    config.sep_token_id = 2
    config.pad_token_id = 0
    config.bos_token_id = 1
    config.eos_token_id = 2

    return config
