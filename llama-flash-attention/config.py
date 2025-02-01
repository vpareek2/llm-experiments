import torch
import numpy as np
import jax.numpy as jnp
from pathlib import Path
from typing import List, NamedTuple

DEFAULT_MASK_VALUE = -0.7 * float(torch.finfo(torch.float32).max)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class LayerWeights(NamedTuple):
    """Weights for a single transformer layer"""

    wq: torch.Tensor
    wk: torch.Tensor
    wv: torch.Tensor
    wo: torch.Tensor
    w1: torch.Tensor
    w2: torch.Tensor
    w3: torch.Tensor
    ffn_norm: torch.Tensor
    attention_norm: torch.Tensor

class LlamaWeights(NamedTuple):
    """Weights for the entire Llama model"""

    tok_embeddings: torch.Tensor
    norm: torch.Tensor
    output: torch.Tensor
    layer_weights: List[LayerWeights]

class ModelParams(NamedTuple):
    """Model hyperparameters for inference"""

    n_layers: int
    n_local_heads: int
    n_local_kv_heads: int
    head_dim: int
    max_seq_len: int
    rope_theta: float
    use_scaled_rope: bool

class SamplerConfig(NamedTuple):
    """Sampling configuration for the model"""

    temperature: float = 0.666
    top_p: float = 0.90
    top_k: int = 27
    min_p: float = 0.03

# LLaMA-1B model hyperparameters
params = {
    "dim": 2048,
    "n_layers": 16,
    "n_heads": 32,
    "n_kv_heads": 8,
    "vocab_size": 128256,
    "ffn_dim_multiplier": 1.5,
    "multiple_of": 256,
    "norm_eps": 1e-05,
    "rope_theta": 500000.0,
    "use_scaled_rope": True,
    "max_seq_len": 4096
}

LLAMA_1B_PARAMS = ModelParams(
    n_layers=params["n_layers"],
    n_local_heads=params["n_heads"],
    n_local_kv_heads=params["n_kv_heads"],
    head_dim=params["dim"] // params["n_heads"],
    max_seq_len=params["max_seq_len"],
    rope_theta=params["rope_theta"],
    use_scaled_rope=params["use_scaled_rope"]
)

def load_weights(ckpt_dir: Path = Path('Llama-3.2-1B-Instruct/'), n_layers: int = 16):
  """Load LLaMA weights from a directory of .npy files"""

  w = {}
  layer_weights = []
  with torch.inference_mode():
    for file in ckpt_dir.glob("*.npy"):
      name = '.'.join(str(file).split('/')[-1].split('.')[:-1])
      jax_weight = jnp.load(file=file, mmap_mode='r', allow_pickle=True)
      np_weight = np.array(jax_weight).astype(np.float32)
      weight = torch.from_numpy(np_weight).to(torch.bfloat16).to(DEVICE)
      w[name] = weight.to(DEVICE)

    for i in range(n_layers):
      layer_weights.append(LayerWeights(
        wq=w[f'layers.{i}.attention.wq.weight'],
        wk=w[f'layers.{i}.attention.wk.weight'],
        wv=w[f'layers.{i}.attention.wv.weight'],
        wo=w[f'layers.{i}.attention.wo.weight'],
        w1=w[f'layers.{i}.feed_forward.w1.weight'],
        w2=w[f'layers.{i}.feed_forward.w2.weight'],
        w3=w[f'layers.{i}.feed_forward.w3.weight'],
        ffn_norm=w[f'layers.{i}.ffn_norm.weight'],
        attention_norm=w[f'layers.{i}.attention_norm.weight'],
      ))

    weights = LlamaWeights(
      tok_embeddings=w['tok_embeddings.weight'],
      norm=w['norm.weight'],
      output=w['output.weight'],
      layer_weights=layer_weights
    )

    return weights