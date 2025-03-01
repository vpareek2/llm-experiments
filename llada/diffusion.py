import jax
import jax.numpy as jnp
import jax.random as jr
from typing import Any, Callable, Dict, Optional, Tuple
from functools import partial

from mask_predictor import predict_masked_tokens

def create_masked_sequence(prompt_ids: jax.Array, response_length: int, mask_token_id: int) -> jax.Array:
    """Create a fully masked sequence for the response"""

    bsz = prompt_ids.shape[0]
    masked_response = jnp.full((bsz, response_length), mask_token_id)
    return jnp.concatenate([prompt_ids, masked_response], axis=1)

def random_remask(key: jax.Array, predicted_ids: jax.Array, current_ids: jax.Array, mask_positions: jax.Array, mask_token_id: int, s_over_t: float) -> jax.Array:
    """Random remasking strategy for diffusion step t to s"""

    # For each predicted token, remask with probability s/t
    remask_prob = jnp.where(mask_positions, s_over_t, 0.0)
    mask_noise = jr.uniform(key, predicted_ids.shape)
    should_remask = mask_noise < remask_prob

    # Apply remasking
    output_ids = jnp.where(should_remask, mask_token_id, predicted_ids)

    # Keep non-masked tokens unchanged from current_ids
    return jnp.where(mask_positions, output_ids, current_ids)

def low_confidence_remask(predicted_ids: jax.Array, prediction_probs: jax.Array, current_ids: jax.Array, mask_positions: jax.Array,  mask_token_id: int, s_over_t: float) -> jax.Array:
    """Low-confidence remasking strategy for diffusion step t to s"""

    bsz, seq_len = current_ids.shape

    # Calculate the number of tokens to keep unmasked
    # s_over_t is the ratio of tokens to remain masked
    # (1 - s_over_t) is the ratio to unmask
    masked_count = jnp.sum(mask_positions, axis=1)
    tokens_to_unmask = ((1 - s_over_t) * masked_count).astype(jnp.int32)

    # Get confidence for each prediction (probability of the selected token)
    confidence = jnp.take_along_axis(prediction_probs, predicted_ids[:, :, None], axis=2).squeeze(-1)

    # For each sequence, get indices of tokens to unmask (highest confidence)
    def get_unmask_indices(sequence_confidence, num_to_unmask):
        return jnp.argsort(sequence_confidence)[-num_to_unmask:]

def semi_autoregressive_remask(predicted_ids: jax.Array, prediction_probs: jax.Array, current_ids: jax.Array, mask_positions: jax.Array, mask_token_id: int, s_over_t: float, block_size: int) -> jax.Array:
    """Semi-autoregressive remasking for left-to-right generation in blocks"""
    bsz, seq_len = current_ids.shape

    # Find the leftmost masked position in each sequence
    # We'll use this to determine the current block
    masked_indices = mask_positions.astype(jnp.int32) * jnp.arange(seq_len)
    leftmost_masked = jnp.argmax(masked_indices, axis=1)

    # Calculate block boundaries
    current_block_start = leftmost_masked
    current_block_end = jnp.minimum(current_block_start + block_size, seq_len)

    # Create mask for positions in the current block
    position_indices = jnp.arange(seq_len)
    in_current_block = (position_indices[None, :] >= current_block_start[:, None]) & \
                        (position_indices[None, :] < current_block_end[:, None]) & \
                        mask_positions

    # Apply low-confidence remasking within the current block
    return low_confidence_remask(
        predicted_ids=predicted_ids,
        prediction_probs=prediction_probs,
        current_ids=current_ids,
        mask_positions=in_current_block,
        mask_token_id=mask_token_id,
        s_over_t=s_over_t
    )

def diffusion_step(key: jax.Array, transformer, lm_head_weights: jax.Array, current_ids: jax.Array, prompt_length: int, mask_token_id: int, t: float, s: float, remask_fn: Callable) -> jax.Array:
    """Perform one step of the reverse diffusion process"""

    # Separate prompt and response
    response_ids = current_ids[:, prompt_length:]

    # Predict tokens
    logits, mask_positions = predict_masked_tokens(
        transformer=transformer,
        input_ids=current_ids,
        lm_head_weights=lm_head_weights,
        mask_token_id=mask_token_id
    )

    # Get response-specific positions
    response_mask = mask_positions[:, prompt_length:]
    response_logits = logits[:, prompt_length:]

    # Get predictions
    prediction_probs = jax.nn.softmax(response_logits, axis=-1)
    predicted_ids = jnp.argmax(response_logits, axis=-1)

    # Apply remasking
    new_response_ids = remask_fn(
        key=key,
        predicted_ids=predicted_ids,
        prediction_probs=prediction_probs,
        current_ids=response_ids,
        mask_positions=response_mask,
        mask_token_id=mask_token_id,
        s_over_t=s/t
    )

    # Recombine with prompt
    return jnp.concatenate([current_ids[:, :prompt_length], new_response_ids], axis=1)

def generate(key: jax.Array, transformer, lm_head_weights: jax.Array, prompt_ids: jax.Array, config: Dict[str, Any]) -> jax.Array:
    """Generate a response using the diffusion process"""

    response_length = config.get('response_length', 128)
    num_steps = config.get('num_steps', 20)
    mask_token_id = config.get('mask_token_id')
    remask_strategy = config.get('remask_strategy', 'random')
    block_size = config.get('block_size', 32)

    # Create initial sequence with masked response
    current_ids = create_masked_sequence(prompt_ids, response_length, mask_token_id)
    prompt_length = prompt_ids.shape[1]

    # Select the remasking function based on strategy
    if remask_strategy == 'random':
        remask_fn = partial(random_remask)
    elif remask_strategy == 'low_confidence':
        remask_fn = partial(low_confidence_remask)
    elif remask_strategy == 'semi_autoregressive':
        remask_fn = partial(semi_autoregressive_remask, block_size=block_size)
    else:
        raise ValueError(f"Unknown remasking strategy: {remask_strategy}")

     # Perform diffusion steps
    t_values = jnp.linspace(1.0, 0.0, num_steps + 1)

    def diffusion_loop(i, current_state):
        current_ids, key = current_state
        key, step_key = jr.split(key)

        t, s = t_values[i], t_values[i+1]
        next_ids = diffusion_step(
            key=step_key,
            transformer=transformer,
            lm_head_weights=lm_head_weights,
            current_ids=current_ids,
            prompt_length=prompt_length,
            mask_token_id=mask_token_id,
            t=t,
            s=s,
            remask_fn=remask_fn
        )

        return next_ids, key

    # Run the diffusion loop
    for i in range(num_steps):
        key, step_key = jr.split(key)
        current_ids = diffusion_step(
            key=step_key,
            transformer=transformer,
            lm_head_weights=lm_head_weights,
            current_ids=current_ids,
            prompt_length=prompt_length,
            mask_token_id=mask_token_id,
            t=t_values[i],
            s=t_values[i+1],
            remask_fn=remask_fn
        )

    return current_ids
