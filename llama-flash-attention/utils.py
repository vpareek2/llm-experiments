import torch
from config import DEVICE

def precompute_freqs_cis(dim: int, end: int, theta: float = 500000.0, use_scaled: bool = False, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """Precompute the roatary frequencies for RoPE."""

    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=dtype, device=DEVICE)[: (dim // 2)] / dim))
    # (Optional) Apply any frequency scaling here if desired.
    t = torch.arange(end, dtype=dtype, device=DEVICE).unsqueeze(1)
    freqs = freqs.unsqueeze(0)  # shape (1, dim//2)
    freqs = t * freqs
    return torch.exp(1j * freqs)

def build_attn_mask(seqlen: int, start_pos: int) -> torch.Tensor:
    """Builds the casual attention mask for the transformer layer."""

    if seqlen > 1:
        mask = torch.full((seqlen, seqlen), float("-inf"), device=DEVICE)
        mask = torch.triu(mask, diagonal=1)
        mask = torch.hstack([torch.zeros((seqlen, start_pos), device=DEVICE), mask])
        return mask.to(torch.float32)
    return None