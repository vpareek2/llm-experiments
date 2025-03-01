from dataclasses import dataclass
import jax

@dataclass
class ModelConfig:
    """Model Configuration Parameters for LLaDa"""

    d_model: int = 4096
    n_layers: int = 32
    n_heads: int = 32
    n_kv_heads: int = 32
    mlp_hidden_size: int = 12288
    vocab_size: int = 126464
    max_sequence_length: int = 4096
    mask_token_id: int = 126336

@dataclass
class FFNWeights:
    w1: jax.Array    # Shape: [d_model, mlp_hidden_size]
    w2: jax.Array    # Shape: [mlp_hidden_size, d_model]
    w3: jax.Array    # Shape: [d_model, mlp_hidden_size]
