import torch
from config import DEVICE, LLAMA_1B_PARAMS, load_weights, SamplerConfig
from utils import build_attn_mask, precompute_freqs_cis
from kvcache import KVCache
from tokenizer import Tokenizer
from model import llama

prompt = "Write a short story about a dragon that is afraid of fire."

def _sample(logits: torch.Tensor, temperature: float, top_p: float) -> torch.Tensor:
    """Sample the next token from the logits using temperature and top-p sampling."""

    # Only use the last token's logits for sampling
    probs = torch.softmax(logits[:, -1] / temperature, dim=-1)
    # Sort probabilities in descending order
    probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)
    probs_sum = torch.cumsum(probs_sort, dim=-1)
    mask = probs_sum - probs_sort > top_p
    probs_sort[mask] = 0.0
    # Renormalize probabilities
    probs_sort = probs_sort / probs_sort.sum(dim=-1, keepdim=True)
    next_token = torch.multinomial(probs_sort, num_samples=1)
    next_token = torch.gather(probs_idx, -1, next_token)
    return next_token

from pathlib import Path
def generate(weights, model_params, tokens):
    cur_pos = 0
    # Convert tokens into a tensor with shape (batch, seqlen)
    tokens = torch.tensor([tokens], dtype=torch.long).to(DEVICE)
    bsz, seqlen = tokens.shape
    attn_mask = build_attn_mask(seqlen, cur_pos)
    # Precompute rotary frequencies (for the entire max sequence length)
    freqs_cis = precompute_freqs_cis(model_params.head_dim, model_params.max_seq_len,
                                     model_params.rope_theta, model_params.use_scaled_rope)
    # Create the key/value cache
    kvcache = KVCache.new(model_params.n_layers, bsz, model_params.max_seq_len,
                          model_params.n_local_kv_heads, model_params.head_dim).to(DEVICE)
    # Run the transformer on the prompt tokens
    logits, kvcache = llama(weights, model_params, tokens, cur_pos, freqs_cis[:seqlen], kvcache, attn_mask=attn_mask)
    # Select the next token as the argmax of the logits
    next_token = torch.argmax(logits[:, -1], dim=-1, keepdim=True).to(torch.int32)
    gen_tokens = next_token
    tokenizer = Tokenizer('tokenizer.model')
    print(tokenizer.decode([next_token.item()]), end='', flush=True)
    cur_pos = seqlen
    # Some stop tokens (for example purposes)
    stop = torch.tensor([128001, 128008, 128009], device=DEVICE, dtype=torch.int32)
    cfg = SamplerConfig()
    # Generation loop (with an arbitrary maximum length of 8192)
    while cur_pos < 8192:
        cur_pos += 1
        logits, kvcache = llama(weights, model_params, next_token, cur_pos, freqs_cis[cur_pos:cur_pos+1], kvcache)
        next_token = _sample(logits, temperature=cfg.temperature, top_p=cfg.top_p)
        gen_tokens = torch.cat((gen_tokens.to(DEVICE), next_token.to(DEVICE)), dim=1)
        out_token = tokenizer.decode(next_token.tolist()[0])
        print(out_token, end='', flush=True)
        if torch.isin(next_token, stop).any():
            break

def main():
    with torch.inference_mode():
        # Load model weights
        model_params = LLAMA_1B_PARAMS
        print(model_params)
        weights = load_weights(Path('Llama-3.2-1B-Instruct'), n_layers=model_params.n_layers)
        print("Loaded weights")
        tokenizer = Tokenizer('tokenizer.model')
        print("Loaded tokenizer")
        raw_tokens = tokenizer.encode(prompt, bos=False, eos=False, allowed_special='all')
        print(prompt)
        generate(weights, model_params, raw_tokens)

if __name__ == '__main__':
    main()