import torch
import torch.nn.functional as F
from torch.optim import AdamW
from typing import List, Callable, Dict, Any
from dataclasses import dataclass

@dataclass
class GRPOConfig:
    """Configuration for GRPO training."""
    learning_rate: float = 5e-6
    num_train_epochs: int = 1
    per_device_train_batch_size: int = 1
    gradient_accumulation_steps: int = 4
    num_generations: int = 16   # Number of completions per prompt.
    max_prompt_length: int = 256
    max_completion_length: int = 786
    max_grad_norm: float = 0.1
    kl_coeff: float = 0.04      # KL penalty coefficient.
    eps: float = 0.2            # Clipping parameter for PPO.

class GRPOTrainer:
    def __init__(
        self,
        model: torch.nn.Module,
        tokenizer: Any,
        reward_funcs: List[Callable],
        train_dataset: Any,
        config: GRPOConfig
    ):
        """
        Args:
            model: The language model to train.
            tokenizer: The tokenizer used for encoding/decoding text.
            reward_funcs: List of callable reward functions. Each function should accept
                          (prompts, completions, answers) and return a list of reward scores.
            train_dataset: The training dataset, assumed to yield batches (dicts) with keys "prompt" and "answer".
            config: GRPOConfig instance.
        """
        self.model = model
        self.tokenizer = tokenizer
        self.reward_funcs = reward_funcs
        self.train_dataset = train_dataset
        self.config = config
        self.device = next(model.parameters()).device

        # Create an optimizer.
        self.optimizer = AdamW(model.parameters(), lr=config.learning_rate)

        # Clone the model to create a reference model.
        # Here we assume that instantiating with model.config is sufficient.
        self.ref_model = type(model)(model.config).to(self.device)
        self.ref_model.load_state_dict(model.state_dict())
        self.ref_model.eval()

        self.steps = 0

    def generate_group(self, prompt_ids: torch.Tensor) -> torch.Tensor:
        """
        Generate multiple responses for each prompt in the batch in one batched call.

        Args:
            prompt_ids: Tensor of shape (batch_size, seq_len).

        Returns:
            Tensor of shape (batch_size, num_generations, generated_seq_len)
        """
        batch_size = prompt_ids.shape[0]
        # Expand each prompt to have `num_generations` copies.
        expanded_prompt_ids = prompt_ids.unsqueeze(1).expand(batch_size, self.config.num_generations, -1)
        expanded_prompt_ids = expanded_prompt_ids.reshape(-1, prompt_ids.shape[-1])

        # Generate completions in a single call.
        with torch.no_grad():
            generated = self.model.generate(
                expanded_prompt_ids,
                max_length=self.config.max_completion_length,
                do_sample=True,
                top_p=0.95,
                temperature=0.7,
                pad_token_id=self.tokenizer.pad_token_id,
            )
        # Reshape to (batch_size, num_generations, seq_len)
        generated = generated.view(batch_size, self.config.num_generations, -1)
        return generated

    def compute_rewards(self, prompts: List[str], completions: List[List[str]], answers: List[str]) -> torch.Tensor:
        """
        Compute rewards for each prompt and its group of completions.

        Args:
            prompts: List of prompt strings (length = batch_size).
            completions: List (length=batch_size) of lists (length=num_generations) of completion strings.
            answers: List of answer strings corresponding to the prompts.

        Returns:
            Tensor of shape (batch_size, num_generations) with reward scores.
        """
        batch_size = len(prompts)
        rewards_all = torch.zeros(batch_size, self.config.num_generations, device=self.device)
        # Process each prompt group.
        for i in range(batch_size):
            prompt_text = prompts[i]
            group_completions = completions[i]
            answer = answers[i]
            group_reward = torch.zeros(self.config.num_generations, device=self.device)
            # Sum rewards from each reward function.
            for reward_func in self.reward_funcs:
                # Expecting reward_func to process a group: it receives lists of repeated prompt and answer.
                rewards = reward_func(
                    prompts=[prompt_text] * self.config.num_generations,
                    completions=group_completions,
                    answers=[answer] * self.config.num_generations
                )
                group_reward += torch.tensor(rewards, dtype=torch.float, device=self.device)
            rewards_all[i] = group_reward
        return rewards_all

    def compute_advantages(self, rewards: torch.Tensor) -> torch.Tensor:
        """
        Compute normalized advantages per prompt group.

        Args:
            rewards: Tensor of shape (batch_size, num_generations)

        Returns:
            Tensor of shape (batch_size, num_generations) with normalized advantages.
        """
        # Compute advantages per prompt group.
        advantages = []
        for group_rewards in rewards:
            mean_reward = group_rewards.mean()
            std_reward = group_rewards.std() if group_rewards.std() > 0 else 1.0
            advantages.append((group_rewards - mean_reward) / std_reward)
        return torch.stack(advantages, dim=0)

    def compute_sequence_log_probs(self, logits: torch.Tensor, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Compute sequence-level log probabilities.
        
        Args:
            logits: Tensor of shape (B, seq_len, vocab_size)
            input_ids: Tensor of shape (B, seq_len) with generated token ids.
            
        Returns:
            Tensor of shape (B,) with summed log probabilities per sequence.
        """
        # Get log probabilities for each token.
        log_probs = F.log_softmax(logits, dim=-1)
        # Gather the log probability for each generated token.
        token_log_probs = log_probs.gather(dim=-1, index=input_ids.unsqueeze(-1)).squeeze(-1)
        # Sum over tokens to get a sequence log probability.
        seq_log_probs = token_log_probs.sum(dim=-1)
        return seq_log_probs

    def train_step(self, batch: Dict[str, Any]) -> float:
        """
        Execute a single training step.

        Args:
            batch: A dictionary with keys "prompt" and "answer".

        Returns:
            The loss value (a float).
        """
        self.model.train()
        total_loss = 0.0

        # Extract prompts and answers.
        # Assume batch['prompt'] and batch['answer'] are lists of strings.
        prompts = batch['prompt']
        answers = batch['answer']

        # Tokenize prompts (with truncation to max_prompt_length).
        prompt_inputs = self.tokenizer(
            prompts,
            return_tensors='pt',
            padding=True,
            truncation=True,
            max_length=self.config.max_prompt_length
        )
        # Move inputs to device.
        prompt_inputs = {k: v.to(self.device) for k, v in prompt_inputs.items()}
        prompt_ids = prompt_inputs['input_ids']

        # Generate a group of completions.
        completions_ids = self.generate_group(prompt_ids)  # shape: (batch_size, num_generations, seq_len)

        # Decode completions.
        batch_size = completions_ids.shape[0]
        # Flatten to decode in batch.
        flat_completion_ids = completions_ids.view(-1, completions_ids.shape[-1])
        flat_decoded = self.tokenizer.batch_decode(flat_completion_ids, skip_special_tokens=True)
        # Reshape back to (batch_size, num_generations)
        completions = [
            [flat_decoded[i * self.config.num_generations + j].strip() for j in range(self.config.num_generations)]
            for i in range(batch_size)
        ]

        # Compute rewards per group.
        rewards_tensor = self.compute_rewards(prompts, completions, answers)  # shape: (batch_size, num_generations)

        # Compute advantages per group.
        advantages = self.compute_advantages(rewards_tensor)  # shape: (batch_size, num_generations)
        flat_advantages = advantages.view(-1)  # shape: (batch_size * num_generations)

        # Evaluate the model and reference model on the generated completions.
        # Here we assume that the generation does not include the prompt context.
        flat_completion_ids = flat_completion_ids.to(self.device)
        outputs = self.model(flat_completion_ids)
        ref_outputs = self.ref_model(flat_completion_ids)
        
        # Compute sequence-level log probabilities.
        policy_seq_log_probs = self.compute_sequence_log_probs(outputs.logits, flat_completion_ids)
        ref_seq_log_probs = self.compute_sequence_log_probs(ref_outputs.logits, flat_completion_ids)
        
        # Compute the probability ratio.
        ratio = torch.exp(policy_seq_log_probs - ref_seq_log_probs)  # shape: (batch_size * num_generations)

        # Compute the PPO surrogate objective.
        surrogate1 = ratio * flat_advantages
        surrogate2 = torch.clamp(ratio, 1 - self.config.eps, 1 + self.config.eps) * flat_advantages
        policy_loss = -torch.min(surrogate1, surrogate2).mean()

        # Compute KL divergence using token-level distributions.
        # We use the built-in kl_div with batchmean reduction.
        kl_div = F.kl_div(
            F.log_softmax(outputs.logits, dim=-1),
            F.softmax(ref_outputs.logits, dim=-1),
            reduction='batchmean'
        )

        # Total loss includes the KL penalty.
        loss = policy_loss + self.config.kl_coeff * kl_div

        # Backward pass with gradient accumulation.
        loss_scaled = loss / self.config.gradient_accumulation_steps
        loss_scaled.backward()
        total_loss += loss.item()

        self.steps += 1
        if self.steps % self.config.gradient_accumulation_steps == 0:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
            self.optimizer.step()
            self.optimizer.zero_grad()

        return total_loss

    def train(self):
        """Train the model for the specified number of epochs."""
        self.model.train()
        for epoch in range(self.config.num_train_epochs):
            for batch_idx, batch in enumerate(self.train_dataset):
                loss = self.train_step(batch)
                if batch_idx % 10 == 0:
                    print(f"Epoch {epoch}, Batch {batch_idx}, Loss: {loss:.4f}")
                # Optionally update the reference model periodically.
                if (batch_idx + 1) % 100 == 0:
                    self.ref_model.load_state_dict(self.model.state_dict())
