from dataclasses import dataclass
from typing import List, Callable, Tuple, Optional
import torch
from config import ModelParams

@dataclass
class GRPOConfig:
    """Configuration for GRPO training"""
    num_generations: int = 16  # G in paper
    kl_coef: float = 0.1      # β coefficient
    learning_rate: float = 5e-6
    clip_range: float = 0.2
    max_length: int = 512
    
@dataclass 
class TrainBatch:
    """Container for batched training data"""
    prompts: List[str]
    answers: Optional[List[str]] = None
    generated: Optional[List[str]] = None
    rewards: Optional[torch.Tensor] = None

def generate_group_samples(
    weights: LlamaWeights,
    model_params: ModelParams,
    tokenizer: Tokenizer,
    prompts: List[str], 
    grpo_config: GRPOConfig,
) -> List[str]:
    """Generate G samples for each prompt using existing generate function"""

def compute_rewards(
    batch: TrainBatch,
    reward_funcs: List[Callable], 
) -> torch.Tensor:
    """Compute and aggregate rewards from multiple functions"""

def compute_group_advantages(
    rewards: torch.Tensor,
    grpo_config: GRPOConfig
) -> torch.Tensor:
    """Compute advantages using group statistics"""

def compute_kl_div(
    policy_logits: torch.Tensor,  
    ref_logits: torch.Tensor,
    kl_coef: float
) -> torch.Tensor:
    """Compute KL divergence penalty"""

def compute_grpo_loss(
    policy_logits: torch.Tensor,
    ref_logits: torch.Tensor,
    advantages: torch.Tensor,
    grpo_config: GRPOConfig
) -> torch.Tensor:
    """Compute core GRPO loss"""

def train_step(
    weights: LlamaWeights,
    ref_weights: LlamaWeights,
    model_params: ModelParams,
    batch: TrainBatch,
    optimizer: torch.optim.Optimizer,
    grpo_config: GRPOConfig
) -> Tuple[torch.Tensor, dict]:
    """Single training step returning loss and metrics"""

def train_grpo(
    weights: LlamaWeights,
    model_params: ModelParams,
    tokenizer: Tokenizer,
    train_data: List[TrainBatch],
    reward_funcs: List[Callable],
    grpo_config: GRPOConfig
) -> LlamaWeights:
    """Main training loop"""