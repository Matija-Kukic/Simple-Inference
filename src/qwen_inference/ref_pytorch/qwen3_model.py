import torch
import torch.nn.functional as F
from torch import bfloat16, nn


class QwenConfig:
    def __init__(self):
        self.rope_theta = 1000000
        self.hidden_size = 2560
        self.num_attention_heads = 32
        self.num_hidden_layers = 36
        self.num_key_value_heads = 8
        self.rms_norm_eps = 1e-06
        self.torch_dtype = bfloat16
        self.head_dim = 128
        self.intermediate_size = 9728


class RoPE(nn.Module):
    def __init__(self, config: QwenConfig) -> None:
        super().__init__()
        self.config = config
        i = torch.arange(0, config.head_dim / 2, 1)
        rotations = (config.rope_theta * 1.0) ** ((-2 * i) / config.head_dim)
        self.rot_emb = nn.Buffer(torch.cat((rotations, rotations), dim=-1))

    def _rotate_half(self, x: torch.Tensor) -> torch.Tensor:
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)

    def forward(
        self,
        position_ids: torch.Tensor,
        q: torch.Tensor,
        k: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        ASSUMES B,S,H,D
        position_ids is (B,S) and rot_emb is D so add 2d to make it B,S,1,D
        """
        angles = position_ids[..., None, None] * self.rot_emb
        cos = torch.cos(angles).to(q.dtype)
        sin = torch.sin(angles).to(q.dtype)
        q_emb = q * cos + self._rotate_half(q) * sin
        k_emb = k * cos + self._rotate_half(k) * sin
        return q_emb, k_emb


class MLP(nn.Module):
    def __init__(self, config: QwenConfig) -> None:
        super().__init__()
        self.config = config
        self.up_proj = nn.Linear(
            config.hidden_size, config.intermediate_size, bias=False
        )
        self.gate_proj = nn.Linear(
            config.hidden_size, config.intermediate_size, bias=False
        )
        self.down_proj = nn.Linear(
            config.intermediate_size, config.hidden_size, bias=False
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class Attention(nn.Module):
    def __init__(self, config: QwenConfig) -> None:
        super().__init__()
        self.config = config
        # BAD
        # self.kv_cache = torch.Tensor()
        self.q_proj = nn.Linear(
            config.hidden_size, config.num_attention_heads * config.head_dim, bias=False
        )
        self.k_proj = nn.Linear(
            config.hidden_size, config.num_key_value_heads * config.head_dim, bias=False
        )
        self.v_proj = nn.Linear(
            config.hidden_size, config.num_key_value_heads * config.head_dim, bias=False
        )
        self.k_norm = nn.RMSNorm([config.head_dim], eps=config.rms_norm_eps)
        self.q_norm = nn.RMSNorm([config.head_dim], eps=config.rms_norm_eps)
        self.o_proj = nn.Linear(
            config.num_attention_heads * config.head_dim, config.hidden_size, bias=False
        )


class DecoderLayer(nn.Module):
    def __init__(self, config: QwenConfig) -> None:
        super().__init__()
        self.config = config
        self.input_layernorm = nn.RMSNorm([config.hidden_size], eps=config.rms_norm_eps)
        self.mlp = MLP(config)
        self.post_attention_layernorm = nn.RMSNorm(
            [config.hidden_size], eps=config.rms_norm_eps
        )
        self.self_attn = Attention(config)


class Qwen3(nn.Module):
    def __init__(self) -> None:
        super().__init__()
