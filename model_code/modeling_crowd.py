import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import PreTrainedModel
from transformers.generation import GenerationMixin
from transformers.modeling_outputs import CausalLMOutput

try:
    from .configuration_crowd import CrowdConfig
except ImportError:
    from configuration_crowd import CrowdConfig


def rotate_half(x):
    h = x.shape[-1] // 2
    return torch.cat((-x[..., h:], x[..., :h]), dim=-1)


class CrowdAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        dim = config.hidden_size

        self.q = nn.Linear(dim, self.num_heads * self.head_dim, bias=False)
        self.k = nn.Linear(dim, self.num_kv_heads * self.head_dim, bias=False)
        self.v = nn.Linear(dim, self.num_kv_heads * self.head_dim, bias=False)
        self.o = nn.Linear(dim, dim, bias=False)

    def forward(self, x, cos, sin):
        B, T, _ = x.shape

        q = self.q(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k(x).view(B, T, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v(x).view(B, T, self.num_kv_heads, self.head_dim).transpose(1, 2)

        q = q * cos[:, :, :T] + rotate_half(q) * sin[:, :, :T]
        k = k * cos[:, :, :T] + rotate_half(k) * sin[:, :, :T]

        k = k.repeat_interleave(self.num_heads // self.num_kv_heads, dim=1)
        v = v.repeat_interleave(self.num_heads // self.num_kv_heads, dim=1)

        # Manual matmul + softmax — bit-identical to the reference script.
        # (Do NOT use F.scaled_dot_product_attention here: it may select
        # flash/eff attention kernels with different rounding and flip
        # close argmax results on undertrained checkpoints.)
        a = q @ k.transpose(-2, -1) / math.sqrt(self.head_dim)
        a = a.masked_fill(
            torch.triu(torch.ones(T, T, device=x.device, dtype=torch.bool), 1),
            float("-inf"),
        )
        a = F.softmax(a, dim=-1)

        return self.o((a @ v).transpose(1, 2).reshape(B, T, -1))


class CrowdMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.w1 = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.w2 = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.w3 = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)

    def forward(self, x):
        return self.w3(F.silu(self.w1(x)) * self.w2(x))


class CrowdBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.hidden_size)
        self.attn = CrowdAttention(config)
        self.ln2 = nn.LayerNorm(config.hidden_size)
        self.mlp = CrowdMLP(config)

    def forward(self, x, cos, sin):
        x = x + self.attn(self.ln1(x), cos, sin)
        x = x + self.mlp(self.ln2(x))
        return x


class CrowdEngram(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.table = nn.Embedding(config.engram_buckets, config.hidden_size)

    def forward(self, ids):
        prev = torch.cat([torch.zeros_like(ids[:, :1]), ids[:, :-1]], dim=1)
        h = (prev * 1000003 + ids) % self.table.num_embeddings
        u, inv = torch.unique(h.flatten(), return_inverse=True)
        x = self.table(u.to(self.table.weight.device))
        return x[inv].view(*ids.shape, -1)


class CrowdPreTrainedModel(PreTrainedModel):
    config_class = CrowdConfig
    base_model_prefix = "crowd"
    main_input_name = "input_ids"
    supports_gradient_checkpointing = False
    _no_split_modules = ["CrowdBlock"]
    _supports_static_cache = False
    _supports_cache_class = False

    def _init_weights(self, module):
        # Weights come from safetensors; nothing to init.
        pass


class CrowdForCausalLM(CrowdPreTrainedModel, GenerationMixin):
    # Dict form (required by modern transformers post_init).
    _tied_weights_keys = {"head.weight": "wte.weight"}

    def __init__(self, config):
        super().__init__(config)

        self.wte = nn.Embedding(config.vocab_size, config.hidden_size)
        self.engram = CrowdEngram(config)
        self.blocks = nn.ModuleList(
            [CrowdBlock(config) for _ in range(config.num_hidden_layers)]
        )
        self.ln = nn.LayerNorm(config.hidden_size)
        self.head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Precompute RoPE exactly like the reference script.
        inv = 1.0 / (
            config.rope_theta
            ** (torch.arange(0, config.head_dim, 2, dtype=torch.float32) / config.head_dim)
        )
        f = torch.outer(torch.arange(config.max_position_embeddings, dtype=torch.float32), inv)
        f = torch.cat((f, f), dim=-1)
        self.register_buffer("cos_cached", f.cos()[None, None], persistent=False)
        self.register_buffer("sin_cached", f.sin()[None, None], persistent=False)

        self.post_init()

    # Embeddings API

    def get_input_embeddings(self):
        return self.wte

    def set_input_embeddings(self, value):
        self.wte = value

    def get_output_embeddings(self):
        return self.head

    def set_output_embeddings(self, value):
        self.head = value

    def _tie_weights(self):
        if getattr(self.config, "tie_word_embeddings", True):
            self.head.weight = self.wte.weight

    # Forward — mirrors the reference script exactly

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        inputs_embeds=None,
        labels=None,
        use_cache=False,
        return_dict=True,
        **kwargs,
    ):
        if inputs_embeds is not None:
            raise NotImplementedError("Crowd-v1 Engram requires input_ids.")
        if input_ids is None:
            raise ValueError("input_ids required")

        T = input_ids.shape[1]
        if T > self.config.max_position_embeddings:
            raise ValueError(
                f"seq_len {T} exceeds max_position_embeddings "
                f"({self.config.max_position_embeddings})"
            )

        x = self.wte(input_ids) + self.engram(input_ids).to(self.wte.weight.dtype)

        cos = self.cos_cached.to(x.dtype)
        sin = self.sin_cached.to(x.dtype)

        for block in self.blocks:
            x = block(x, cos, sin)

        logits = self.head(self.ln(x))

        loss = None
        if labels is not None:
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous().to(logits.device)
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100,
            )

        if not return_dict:
            out = (logits,)
            return (loss,) + out if loss is not None else out

        return CausalLMOutput(loss=loss, logits=logits)

    def prepare_inputs_for_generation(self, input_ids, attention_mask=None, **kwargs):
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "use_cache": False,
        }