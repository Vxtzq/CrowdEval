from transformers import PretrainedConfig


class CrowdConfig(PretrainedConfig):
    model_type = "crowd"

    def __init__(
        self,
        vocab_size=151669,
        hidden_size=1536,
        num_hidden_layers=24,
        num_attention_heads=16,
        num_key_value_heads=4,
        head_dim=96,
        intermediate_size=2560,
        max_position_embeddings=2048,
        hidden_act="silu",
        layer_norm_eps=1e-5,
        rope_theta=10000.0,
        attention_bias=False,
        mlp_bias=False,
        tie_word_embeddings=True,
        engram_buckets=227865,
        initializer_range=0.02,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim
        self.intermediate_size = intermediate_size
        self.max_position_embeddings = max_position_embeddings
        self.hidden_act = hidden_act
        self.layer_norm_eps = layer_norm_eps
        self.rope_theta = rope_theta
        self.attention_bias = attention_bias
        self.mlp_bias = mlp_bias
        self.engram_buckets = engram_buckets
        self.initializer_range = initializer_range

        super().__init__(tie_word_embeddings=tie_word_embeddings, **kwargs)