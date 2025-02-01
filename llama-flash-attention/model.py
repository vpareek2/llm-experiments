import torch
import torch.nn.functional as F
import math
from typing import Tuple, Optional
from config import LayerWeights, LlamaWeights, ModelParams
from kvcache import KVCache
from flash_attention import TritonAttention

def rms_norm(x: torch.Tensor, w: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Root mean square layer normalization"""

    return w * (x * torch.rsqrt(torch.pow(x, 2).mean(-1, keepdim=True) + eps))

def apply_rotary_emb(xq: torch.Tensor, xk: torch.Tensor, freqs_cis: torch.Tensor, dtype: torch.dtype = torch.float32) -> Tuple[torch.Tensor, torch.Tensor]:
    """Rotary Positional Embeddings"""

    reshape_xq = xq.float().reshape(*xq.shape[:-1], -1, 2)
    reshape_xk = xk.float().reshape(*xk.shape[:-1], -1, 2)
    xq_ = torch.complex(reshape_xq[..., 0], reshape_xq[..., 1])
    xk_ = torch.complex(reshape_xk[..., 0], reshape_xk[..., 1])
    xq_out = xq_ * freqs_cis.unsqueeze(0).unsqueeze(2)
    xk_out = xk_ * freqs_cis.unsqueeze(0).unsqueeze(2)
    xq_out = torch.stack((xq_out.real, xq_out.imag), dim=-1).reshape(*xq_out.shape[:-1], -1)
    xk_out = torch.stack((xk_out.real, xk_out.imag), dim=-1).reshape(*xk_out.shape[:-1], -1)
    return xq_out.to(dtype), xk_out.to(dtype)

def flash_attention(x: torch.Tensor, layer_weights: LayerWeights, model_params: ModelParams, cur_pos: int, layer_idx: int, freqs_cis: torch.Tensor, kvcache: KVCache, attn_mask: Optional[torch.Tensor] = None,) -> Tuple[torch.Tensor, KVCache]:
    """Flash Attention integrated with Grouped Query Attention"""
    bsz = x.shape[0]
    n_rep = model_params.n_local_heads // model_params.n_local_kv_heads

    xq = F.linear(x, layer_weights.wq).reshape(bsz, -1, model_params.n_local_heads, model_params.head_dim)
    xk = F.linear(x, layer_weights.wk).reshape(bsz, -1, model_params.n_local_kv_heads, model_params.head_dim)
    xv = F.linear(x, layer_weights.wv).reshape(bsz, -1, model_params.n_local_kv_heads, model_params.head_dim)

    xq, xk = apply_rotary_emb(xq, xk, freqs_cis=freqs_cis)
    keys, values, kvcache = kvcache.update(xk, xv, layer_idx, cur_pos, n_rep)

    q = torch.permute(xq, (0, 2, 1, 3))
    k = torch.permute(keys, (0, 2, 1, 3))
    v = torch.permute(values, (0, 2, 1, 3))

    softmax_scale = 1 / math.sqrt(model_params.head_dim)
    casual_flag = attn_mask is not None

    q = q.to(torch.float16)
    k = k.to(torch.float16)
    v = v.to(torch.float16)

    with torch.no_grad():
        flash_out = TritonAttention.apply(q, k, v, casual_flag, softmax_scale)
    
    attn_out = flash_out.transpose(1, 2).reshape(bsz, -1, model_params.n_local_heads * model_params.head_dim)
    out = F.linear(attn_out, layer_weights.wo)

    return out, kvcache

def feed_forward(x: torch.Tensor, layer_weights: LayerWeights) -> torch.Tensor:
    """Feed forward layer"""

    return F.linear(F.silu(F.linear(x, layer_weights.w1)) * F.linear(x, layer_weights.w3), layer_weights.w2)

def llama(weights: LlamaWeights, model_params: ModelParams, tokens: torch.Tensor, cur_pos: int, freqs_cis: torch.Tensor, kvcache: KVCache, attn_mask: Optional[torch.Tensor]=None) -> Tuple[torch.Tensor, KVCache]:
    h = weights.tok_embeddings[tokens]
    for i in range(model_params.n_layers):
        norm_x = rms_norm(h, weights.layer_weights[i].attention_norm)
        h_attn, kvcache = flash_attention(norm_x, weights.layer_weights[i], model_params, cur_pos, i, freqs_cis, kvcache, attn_mask=attn_mask)
        h = h + h_attn
        h = h + feed_forward(rms_norm(h, weights.layer_weights[i].ffn_norm), weights.layer_weights[i])
    logits = F.linear(rms_norm(h, weights.norm), weights.output)
    return logits, kvcache