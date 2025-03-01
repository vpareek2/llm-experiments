import jax
import jax.numpy as jnp
import math
from typing import Dict, Optional, Tuple

from config import FFNWeights, ModelConfig

def rms_norm(x: jax.Array, weight: jax.Array, eps: float = 1e-5) -> jax.Array:
    """Apply Root Mean Squared Normalization"""

    variance = jnp.mean(jnp.square(x), axis=-1, keepdims=True)
    x = x * jax.lax.rsqrt(variance + eps)
    return x * weight

def feed_forward(x: jax.Array, weights: FFNWeights) -> jax.Array:
    """Feed-forward neural network with SwiGLU activation"""

    # SwiGLU activation: (x * w1) * SiLU(x * w3)
    hidden_states = jnp.dot(x, weights.w1)
    gate = jnp.dot(x, weights.w3)
    silu_gate = gate * jax.nn.sigmoid(gate)
    intermediate = hidden_states * silu_gate

    return jnp.dot(intermediate, weights.w2)

def rope(q: jax.Array, k: jax.Array, positions: jax.Array, theta: float = 500000.0, dtype: jnp.dtype = jnp.float32) -> Tuple[jax.Array, jax.Array]:
    """Apply Rotary Positional Embeddings"""

    # Calculate position embeddings
    dim = q.shape[-1] // 2
    inv_freq = 1.0 / (theta ** (jnp.arange(0, dim, dtype=jnp.float32) / dim))

    # Create complex embeddings [seq_len, dim]
    freqs = positions[:, None] * inv_freq[None, :]
    freqs_cis = jnp.exp(1j * freqs)

    # Reshape queries and keys to complex numbers
    reshape_q = q.astype(jnp.float32).reshape(*q.shape[:-1], -1, 2)
    reshape_k = k.astype(jnp.float32).reshape(*k.shape[:-1], -1, 2)
    q_ = jax.lax.complex(reshape_q[..., 0], reshape_q[..., 1])
    k_ = jax.lax.complex(reshape_k[..., 0], reshape_k[..., 1])

    # Apply rotation
    q_out = q_ * freqs_cis[None, :, None, :]
    k_out = k_ * freqs_cis[None, :, None, :]

    # Convert back to real
    q_out = jnp.stack((jnp.real(q_out), jnp.imag(q_out)), axis=-1).reshape(*q_out.shape[:-1], -1)
    k_out = jnp.stack((jnp.real(k_out), jnp.imag(k_out)), axis=-1).reshape(*k_out.shape[:-1], -1)

    return q_out.astype(dtype), k_out.astype(dtype)

def attention(q: jax.Array, k: jax.Array, v: jax.Array, mask: Optional[jax.Array] = None) -> jax.Array:
    """Multi-Head Attention"""

    # Compute Attention scores - shape: [batch, n_heads, seq_len, seq_len]
    scores = jnp.einsum('bthd,bshd->bhts', q, k)

    # Scale the scores
    head_dim = q.shape[-1]
    scores = scores / math.sqrt(head_dim)

    # Apply mask if provided (for masked toks)
    if mask is not None:
        # Shape for broadcasting
        if mask.ndim == 2:
            # [batch, seq_len] -> [batch, 1, 1, seq_len]
            mask = mask[:, None, None, :]

        # Apply large negative value to masked positions
        scores = jnp.where(mask == 0, -1e9, scores)

    # Softmax (logits -> probs)
    weights = jax.nn.softmax(scores, axis=-1)

    # Apply attention weights to values
    output = jnp.einsum('bhts,bshd->bthd', weights, v)

    return output

def transformer_block(x: jax.Array, attn_weights: Dict[str, jax.Array], ffn_weights: FFNWeights, norm_weights: Dict[str, jax.Array], positions: jax.Array, mask: Optional[jax.Array] = None) -> jax.Array:
    """Full LLaDa transformer block"""

    # Layernorm before attention
    norm_x = rms_norm(x, norm_weights['attn_norm'])

    # Project to q, k, v
    bsz, seq_len, d_model = x.shape
    n_heads = attn_weights['q_proj'].shape[1]
    head_dim = d_model // n_heads

    q = jnp.einsum('bld,dnh->blnh', norm_x, attn_weights['q_proj'])
    k = jnp.einsum('bld,dnh->blnh', norm_x, attn_weights['k_proj'])
    v = jnp.einsum('bld,dnh->blnh', norm_x, attn_weights['v_proj'])

    # Apply RoPE
    q_rot, k_rot = rope (q, k, positions)

    # Compute Attention
    attn_output = attention(q_rot, k_rot, v, mask)

    # Project back to d_model
    attn_output = jnp.einsum('blnh,nhd->bld', attn_output, attn_weights['o_proj'])

    # Residual connection
    x = x + attn_output

    # Layernorm before FFN
    norm_x = rms_norm(x, norm_weights['ffn_norm'])

    # FFN
    ffn_output = feed_forward(norm_x, ffn_weights)

    # Residual connection
    x = x + ffn_output

    return x

class Transformer:

    def __init__(self, config: ModelConfig, weights=None):
        self.config = config
        self.weights = weights
        # If weights is none, initialize here

    def __call__(self, input_ids: jax.Array, mask: Optional[jax.Array] = None) -> jax.Array:
        """Forward pass through the transformer"""

        # Position indices
        seq_len = input_ids.shape[1]
        positions = jnp.arange(seq_len)

        # Token Embeddings
        x = self.weights['token_embedding'][input_ids]

        # Apply transformer blocks
        for i in range(self.config.n_layers):
            layer_weights = {
                'attn_weights': {
                    'q_proj': self.weights[f'layer_{i}.q_proj'],
                    'k_proj': self.weights[f'layer_{i}.k_proj'],
                    'v_proj': self.weights[f'layer_{i}.v_proj'],
                    'o_proj': self.weights[f'layer_{i}.o_proj'],
                },
                'ffn_weights': FFNWeights(
                    w1=self.weights[f'layer_{i}.ffn.w1'],
                    w2=self.weights[f'layer_{i}.ffn.w2'],
                    w3=self.weights[f'layer_{i}.ffn.w3'],
                ),
                'norm_weights': {
                    'attn_norm': self.weights[f'layer_{i}.attn_norm'],
                    'ffn_norm': self.weights[f'layer_{i}.ffn_norm'],
                }
            }

            x = transformer_block(x,
                layer_weights['attn_weights'],
                layer_weights['ffn_weights'],
                layer_weights['norm_weights'],
                positions,
                mask
            )

        # Final layernorm
        x = rms_norm(x, self.weights['final_norm'])

        return x
