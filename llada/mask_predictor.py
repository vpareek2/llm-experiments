import jax
import jax.numpy as jnp
from typing import Dict, Optional, Tuple, List
from config import ModelConfig

def get_mask_positions(input_ids: jax.Array, mask_token_id: int) -> jax.Array:
    """Return a boolean mask where tokens are masked"""

    return input_ids == mask_token_id

def predict_tokens(hidden_states: jax.Array, lm_head_weights: jax.Array) -> jax.Array:
    """Predict hidden states to vocabulary  logits"""

    return jnp.einsum('bld,dv->blv', hidden_states, lm_head_weights)

def predict_masked_tokens(transformer, input_ids: jax.Array, lm_head_weights: jax.Array, mask_token_id: int, attention_mask: Optional[jax.Array] = None) -> Tuple[jax.Array, jax.Array]:
    """Generate predictions for masked tokens"""

    # Get transformer hidden states
    hidden_states = transformer(input_ids, attention_mask)

    # Project to vocabulary
    logits = predict_tokens(hidden_states, lm_head_weights)

    # Get mask positions
    mask_positions = get_mask_positions(input_ids, mask_token_id)

    return logits, mask_positions
