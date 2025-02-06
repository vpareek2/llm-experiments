import torch
import torch.nn.functional as F
from torch.optim import AdamW
from typing import List, Callable, Optional, Dict, Any
from dataclasses import dataclass

@dataclass
class GRPOConfig:
    """Configuration for GRPO"""

    learning_rate: float = 5e-6
    num_train_epochs: int = 1
    per_device_train_batch_size: int = 1
    gradient_accumulation_steps: int = 4
    num_generations: int = 16   # Group size for sampling
    max_prompt_length: int = 256
    max_completion_length: int = 786
    max_grad_norm: float = 0.1
    kl_coeff: float = 0.04 # KL penalty coefficient
    eps: float = 0.2 # Clipping parameter

class GRPOTrainer:
    def __init__(self, model: torch.nn.Module, tokenizer: Any, reward_funcs: List[Callable], train_dataset: Any, config: GRPOConfig):
        self.model = model
        self.tokenizer = tokenizer
        self.reward_funcs = reward_funcs
        self.train_dataset = train_dataset
        self.config = config

        self.optimizer = AdamW(model.parameters(), lr=config.learning_rate)

        self.ref_model = type(model)(model.config).to(model.device)
        self.ref_model.load_state_dict(model.state_dict())
        self.ref_model.eval()
    
    def generate_group(self, prompt_ids: torch.Tensor) -> List[torch.Tensor]:
        """Generate multiple responses for each prompt"""

        outputs = []

        for _ in range(self.config.num_generations):
            with torch.no_grad():
                output = self.model.generate(
                    prompt_ids,
                    max_length=self.config.max_completion_length,
                    do_sample=True,
                    top_p=0.95,
                    temperature=0.7,
                    pad_token_id=self.tokenizer.pad_token_id,
                )
                outputs.append(output)

        return outputs
    
    def compute_rewards(self, prompts: List[dict], completions: List[dict], answers: List[str]) -> torch.Tensor:
        """Compute combined rewards from all reward functions"""
        
        total_rewards = torch.zeros(len(completions))

        for reward_func in self.reward_funcs:
            rewards = reward_func(prompts=prompts, completions=completions, answers=answers)
            total_rewards += torch.tensor(rewards, dtype=torch.float)

        return total_rewards
    
    def compute_advantages(self, rewards: torch.Tensor) -> torch.Tensor:
        """Compute advantages using group normalization"""

        mean_reward = rewards.mean()
        std_reward = rewards.std()
        if std_reward == 0:
            std_reward = 1
        
        return (rewards - mean_reward) / std_reward
    
    def compute_kl_divergence(self, policy_logits: torch.Tensor, ref_logits: torch.Tensor) -> torch.Tensor:
        """Compute KL divergence between policy and reference model"""
        
        policy_probs = F.softmax(policy_logits, dim=-1)
        ref_probs = F.softmax(ref_logits, dim=-1)

        # Using unbaised estimator from paper
        ratio = ref_probs / (policy_probs + 1e-8)
        kl = ratio - torch.log(ratio) - 1
        return kl.mean()

    def train_step(self, batch: Dict[str, Any]) -> float:
        """Execute a single training step"""

        self.model.train()
        total_loss = 0

        # Get prompts and answers
        prompts = batch['prompt']
        answers = batch['answer']

        # Tokenize prompts
        prompt_ids = self.tokenizer(prompts, return_tensors='pt', padding=True, truncation=True).to(self.model.device)
        
        # Generate group of responses
        completion_ids = self.generate_group(prompt_ids['input_ids'])

        # Decode completions
        completions = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in completion_ids]

        # Compute rewards and advantages
        rewards = self.compute_rewards(prompts, completions, answers)
        advantages = self.compute_advantages(rewards)

        # Forward pass for policy and reference logits
        for i, completion_id in enumerate(completion_ids):
            outputs = self.model(completion_id)
            ref_outputs = self.ref_model(completion_id)

            policy_logits = outputs.logits
            ref_logits = ref_outputs.logits

            # Compute KL divergence
            kl_div = self.compute_kl_divergence(policy_logits, ref_logits)

            # Compute policy ratio and clipped objective
            log_probs = F.log_softmax(policy_logits, dim=-1)
            old_log_probs = F.log_softmax(ref_logits, dim=-1)
            ratio = torch.exp(log_probs - old_log_probs)

            advantage = advantages[i]

            # Compute clipped objective
            obj1 = ratio * advantages
            obj2 = torch.clamp(ratio, 1 - self.config.eps, 1 + self.config.eps) * advantage
            obj = -torch.min(obj1, obj2).mean()

            # Add KL penalty
            loss = obj + self.config.kl_coeff * kl_div

            # Scale loss and backward
            scaled_loss = loss / self.config.gradient_accumulation_steps
            scaled_loss.backward()

            total_loss += loss.item()
        
        # Gradient clipping and optimizer step
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)

        if (self.steps + 1) % self.config.gradient_accumulation_steps == 0:
            self.optimizer.step()
            self.optimizer.zero_grad()
        
        return total_loss / len(completion_ids)
    
    def train(self):
        """Train the model"""

        self.steps = 0
        self.model.train()

        for epoch in range(self.config.num_train_epochs):
            for batch_idx, batch in enumerate(self.train_dataset):
                loss = self.train_step(batch)

                if batch_idx % 10 == 0:
                    print(f"Epoch {epoch}, Batch {batch_idx}, Loss: {loss:.4f}")
                
                self.steps += 1