# kvcache.py
import torch
import torch.nn as nn
from config import DEVICE

class KVCache(nn.Module):
    def __init__(self, layers: int, bsz: int, max_seq_len: int, kv_heads: int, head_dim: int):
        super(KVCache, self).__init__()
        self.register_buffer(
            'k',
            torch.zeros((layers, bsz, max_seq_len, kv_heads, head_dim), dtype=torch.bfloat16, device=DEVICE)
        )
        self.register_buffer(
            'v',
            torch.zeros((layers, bsz, max_seq_len, kv_heads, head_dim), dtype=torch.bfloat16, device=DEVICE)
        )

    @classmethod
    def new(cls, layers: int, bsz: int, max_seq_len: int, kv_heads: int, head_dim: int) -> 'KVCache':
        return cls(layers, bsz, max_seq_len, kv_heads, head_dim)

    def update(self, xk: torch.Tensor, xv: torch.Tensor, layer_idx: int, cur_pos: int, n_rep: int):
        # Ensure xk and xv have the correct dtype
        xk = xk.to(self.k.dtype)
        xv = xv.to(self.v.dtype)
        insert_len = xk.size(1)
        self.k[layer_idx, :, cur_pos:cur_pos+insert_len, :, :] = xk
        self.v[layer_idx, :, cur_pos:cur_pos+insert_len, :, :] = xv

        # If we are at the first token, repeat the KV projections n_rep times.
        if cur_pos == 0:
            keys = xk.repeat_interleave(n_rep, dim=2)
            values = xv.repeat_interleave(n_rep, dim=2)
        else:
            keys = self.k[layer_idx].repeat_interleave(n_rep, dim=2)
            values = self.v[layer_idx].repeat_interleave(n_rep, dim=2)

        return keys, values, self
