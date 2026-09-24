import torch
import torch.nn.functional as F
from torch import float32, nn


class QwenConfig:
    def __init__(self):
        self.rope_theta = 1000000
        self.hidden_size = 2560
        self.num_attention_heads = 32
        self.head_dim = 128


class RoPE(nn.Module):
    def __init__(self, config: QwenConfig) -> None:
        super().__init__()
        self.config = config
        i = torch.arange(0, config.head_dim / 2, 1)
        rotations = (config.rope_theta * 1.0) ** ((-2 * i) / config.head_dim)
        self.rot_emb = nn.Buffer(torch.cat((rotations, rotations), dim=-1))

    def _rotate_half(self, x: torch.Tensor):
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)

    def forward(
        self,
        position_ids: torch.Tensor,
        q: torch.Tensor,
        k: torch.Tensor,
    ):
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



class Qwen3(nn.Module):
    def __init__(self) -> None:
        super().__init__()
