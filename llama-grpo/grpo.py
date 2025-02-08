import torch
from copy import deepcopy
from dataclasses import dataclass
from collections import defaultdict
from tqdm import tqdm
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, get_cosine_schedule_with_warmup
from typing import Tuple
# ------------------------------------------------------------------------
# CONFIG
# ------------------------------------------------------------------------
@dataclass
class GRPOConfig:
    # Generation (policy) parameters
    num_generations: int = 8           # G in the paper (number of completions per prompt)
    max_length: int = 128              # Maximum sequence length for completions
    temperature: float = 0.7           # Sampling temperature

    # Model and training parameters
    model_name: str = "meta-llama/Llama-3.2-1B"  # Or any causal LM you have access to
    output_dir: str = "outputs/"
    learning_rate: float = 5e-6
    adam_beta1: float = 0.9
    adam_beta2: float = 0.99
    weight_decay: float = 0.1
    warmup_ratio: float = 0.1
    beta: float = 0.04               # KL penalty coefficient
    num_epochs: int = 1
    policy_update_steps: int = 10    # How often to update the "old" (baseline) policy

    # Device
    device: str = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")

# Create a config instance
config = GRPOConfig()

# ------------------------------------------------------------------------
# HELPER FUNCTIONS & REWARD FUNCTIONS
# ------------------------------------------------------------------------
def format_prompt(prompt: str) -> str:
    """
    Optionally add XML tags or other formatting to the prompt.
    (In our example we assume that the prompt is plain text and we add a reasoning/answer template.)
    """
    if not prompt.startswith("<reasoning>"):
        return f"<reasoning>\n{prompt}\n</reasoning>\n<answer>\n"
    return prompt

def extract_xml_answer(text: str) -> str:
    """Extract the text between <answer> and </answer> tags."""
    if "<answer>" not in text or "</answer>" not in text:
        return text
    return text.split("<answer>")[-1].split("</answer>")[0].strip()

# Example reward functions
def correctness_reward_func(prompts, completions, answers, **kwargs) -> list:
    """
    Returns a reward of 2.0 if the extracted answer exactly matches the gold answer,
    and 0.0 otherwise.
    """
    # For simplicity, we assume that prompts and answers are lists of strings,
    # and that completions is a list of lists (each group of completions for one prompt).
    rewards = []
    for prompt, group, answer in zip(prompts, completions, answers):
        # Take the first completion in the group (or you can combine information from all completions)
        response = group[0]
        extracted = extract_xml_answer(response)
        rewards.append(2.0 if extracted == answer.strip() else 0.0)
    return rewards

def soft_format_reward_func(completions, **kwargs) -> list:
    """
    Reward 0.5 if the completion contains both <reasoning> and <answer> tags.
    """
    rewards = []
    for group in completions:
        # Use the first completion as representative
        text = group[0]
        if "<reasoning>" in text and "<answer>" in text:
            rewards.append(0.5)
        else:
            rewards.append(0.0)
    return rewards

# You can combine several reward functions (here we sum two reward signals)
reward_functions = [correctness_reward_func, soft_format_reward_func]

# ------------------------------------------------------------------------
# GRPO CORE FUNCTIONS
# ------------------------------------------------------------------------
def sample_group(prompt: str, model: AutoModelForCausalLM, tokenizer: AutoTokenizer, config: GRPOConfig) -> list:
    """
    Generate a group (list) of completions for a single prompt.
    This corresponds to sampling G completions per prompt.
    """
    formatted_prompt = format_prompt(prompt)
    inputs = tokenizer(formatted_prompt, return_tensors="pt").to(config.device)
    completions = []
    for _ in range(config.num_generations):
        outputs = model.generate(
            **inputs,
            max_length=config.max_length,
            temperature=config.temperature,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
        text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        completions.append(text)
    return completions

def calculate_group_advantages(rewards: torch.Tensor, group_size: int) -> torch.Tensor:
    """
    Compute normalized advantages per group.
    For each prompt group (of size G), we subtract the group mean and divide by the group std.
    """
    # Assume rewards is a tensor of shape (B * G,)
    grouped = rewards.view(-1, group_size)
    group_mean = grouped.mean(dim=1, keepdim=True)
    group_std = grouped.std(dim=1, keepdim=True)
    advantages = (grouped - group_mean) / (group_std + 1e-8)
    return advantages.view(-1)

def calculate_kl_divergence(current_logits: torch.Tensor, ref_logits: torch.Tensor) -> torch.Tensor:
    """
    Compute a token-wise KL divergence (using an unbiased estimator).
    Here we use a simplified formulation:
        KL = exp(ref - current) - (ref - current) - 1
    """
    ratio = torch.exp(ref_logits - current_logits)
    kl = ratio - (ref_logits - current_logits) - 1.0
    return kl

def calculate_policy_ratio(current_logits: torch.Tensor, old_logits: torch.Tensor) -> torch.Tensor:
    """
    Compute the ratio between current and old policy probabilities.
    (This is done in log space and then exponentiated.)
    """
    return torch.exp(current_logits - old_logits)

def calculate_grpo_loss(policy_ratio: torch.Tensor,
                        advantages: torch.Tensor,
                        kl_div: torch.Tensor,
                        beta: float = 0.04,
                        epsilon: float = 0.2,
                        mask: torch.Tensor = None) -> torch.Tensor:
    """
    Compute the GRPO loss, which (like PPO) uses a clipped objective plus a KL penalty.
    """
    # Add dimension to advantages to match policy_ratio shape
    advantages = advantages.unsqueeze(-1)
    
    ratio_clipped = torch.clamp(policy_ratio, 1 - epsilon, 1 + epsilon)
    pg_loss1 = policy_ratio * advantages
    pg_loss2 = ratio_clipped * advantages
    pg_loss = -torch.min(pg_loss1, pg_loss2)  # Negative sign because we want to maximize reward
    loss = pg_loss + beta * kl_div
    if mask is not None:
        loss = (loss * mask).sum() / mask.sum()
    else:
        loss = loss.mean()
    return loss

# ------------------------------------------------------------------------
# SINGLE GRPO TRAINING STEP
# ------------------------------------------------------------------------
def grpo_training_step(batch_prompts: list, batch_answers: list,  # batch_answers is a list of gold answers
                       current_model: AutoModelForCausalLM,
                       old_model: AutoModelForCausalLM,
                       ref_model: AutoModelForCausalLM,
                       reward_functions: list,
                       tokenizer: AutoTokenizer,
                       config: GRPOConfig) -> Tuple[torch.Tensor, dict]:
    """
    Run one GRPO training step for a batch of prompts.
    For each prompt in the batch, sample G completions, compute rewards, and then compute the loss.
    """
    # 1. Generate groups of completions for each prompt in the batch.
    # completions_group is a list (length batch_size) of lists (each of length G).
    completions_group = [sample_group(prompt, current_model, tokenizer, config) for prompt in batch_prompts]

    # 2. Compute rewards for each prompt group.
    # For each reward function, call it (it may use the completions and the gold answers).
    rewards_list = []
    for func in reward_functions:
        # Some reward functions expect the list of prompts and answers (e.g. correctness_reward_func)
        if func.__name__ == "correctness_reward_func":
            rewards = func(batch_prompts, completions_group, batch_answers)
        else:
            rewards = func(completions=completions_group)
        rewards_list.append(torch.tensor(rewards, device=config.device, dtype=torch.float))
    # Sum rewards from all functions (each reward function returns one reward per prompt)
    rewards = torch.stack(rewards_list, dim=1).sum(dim=1)  # shape (batch_size,)

    # 3. Since each prompt had G completions, we need to "expand" rewards to the per-completion level.
    rewards_expanded = rewards.unsqueeze(1).expand(-1, config.num_generations).reshape(-1)

    # 4. Compute group-normalized advantages.
    advantages = calculate_group_advantages(rewards_expanded, config.num_generations)

    # 5. Prepare inputs for computing logits.
    # (For simplicity, we re-tokenize one of the completions per group; in practice you might do this token-wise.)
    # Here we flatten all completions (each group becomes a separate example).
    all_completions = [completion for group in completions_group for completion in group]
    inputs = tokenizer(all_completions, return_tensors="pt", padding=True, truncation=True).to(config.device)

    # 6. Get logits from the current, old, and reference models.
    with torch.no_grad():
        old_outputs = old_model(**inputs)
        ref_outputs = ref_model(**inputs)
    current_outputs = current_model(**inputs)

    # For simplicity we compute a "logit" per example by taking the log probability of the last generated token.
    # (In a more detailed implementation you would compute a per-token loss.)
    current_logits = current_outputs.logits[:, -1].squeeze(-1)
    old_logits = old_outputs.logits[:, -1].squeeze(-1)
    ref_logits = ref_outputs.logits[:, -1].squeeze(-1)

    # 7. Compute policy ratio and KL divergence.
    policy_ratio = calculate_policy_ratio(current_logits, old_logits)
    kl_div = calculate_kl_divergence(current_logits, ref_logits)

    # 8. Compute GRPO loss.
    loss = calculate_grpo_loss(policy_ratio, advantages, kl_div, beta=config.beta)

    # 9. Collect some metrics for logging.
    metrics = {
        "loss": loss.item(),
        "mean_reward": rewards.mean().item(),
        "mean_advantage": advantages.mean().item(),
        "mean_kl": kl_div.mean().item(),
        "mean_policy_ratio": policy_ratio.mean().item()
    }
    return loss, metrics

# ------------------------------------------------------------------------
# TRAINING LOOP
# ------------------------------------------------------------------------
def train_grpo(model: AutoModelForCausalLM,
               train_dataset: Dataset,
               tokenizer: AutoTokenizer,
               config: GRPOConfig,
               reward_functions: list) -> AutoModelForCausalLM:
    """
    Train the model using GRPO.
    
    This simplified training loop:
      - Creates copies of the model to serve as the reference and old (baseline) policies.
      - Iterates over the training dataset (assumed to be a dataset of prompts and corresponding gold answers).
      - Updates the model with the GRPO loss.
    """
    # Make a copy of the model to serve as the reference and old policies.
    ref_model = deepcopy(model)
    old_model = deepcopy(model)
    for param in ref_model.parameters():
        param.requires_grad = False
    for param in old_model.parameters():
        param.requires_grad = False

    model.to(config.device)
    ref_model.to(config.device)
    old_model.to(config.device)
    
    optimizer = torch.optim.AdamW(model.parameters(),
                                  lr=config.learning_rate,
                                  betas=(config.adam_beta1, config.adam_beta2),
                                  weight_decay=config.weight_decay)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(config.warmup_ratio * len(train_dataset)),
        num_training_steps=len(train_dataset) * config.num_epochs
    )
    
    model.train()
    global_step = 0
    best_loss = float("inf")
    for epoch in range(config.num_epochs):
        epoch_metrics = defaultdict(list)
        # Assume that each example in the dataset is a dict with keys "prompt" and "answer"
        for example in tqdm(train_dataset, desc=f"Epoch {epoch}"):
            # For simplicity, here each batch is one example (prompt + gold answer)
            prompt = example["prompt"]
            answer = example.get("answer", "")  # gold answer (if available)
            loss, metrics = grpo_training_step(
                batch_prompts=[prompt],
                batch_answers=[answer],
                current_model=model,
                old_model=old_model,
                ref_model=ref_model,
                reward_functions=reward_functions,
                tokenizer=tokenizer,
                config=config
            )
            loss.backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            # Periodically update the old policy (the baseline) with the current model's weights.
            if global_step % config.policy_update_steps == 0:
                old_model.load_state_dict(model.state_dict())

            for key, value in metrics.items():
                epoch_metrics[key].append(value)
            global_step += 1

            # (Optional) Save the best model checkpoint.
            if loss.item() < best_loss:
                best_loss = loss.item()
                torch.save(model.state_dict(), f"{config.output_dir}/best_model.pt")

        # Log averaged metrics for the epoch.
        print(f"Epoch {epoch} metrics:")
        for key, values in epoch_metrics.items():
            avg_value = sum(values) / len(values)
            print(f"  {key}: {avg_value:.4f}")
    return model

# ------------------------------------------------------------------------
# EXAMPLE USAGE
# ------------------------------------------------------------------------
if __name__ == "__main__":
    # Load (or create) a simple dataset.
    # For demonstration we use a small portion of a dataset from Hugging Face.
    # We expect each example to have a "prompt" (the question) and an "answer" (the gold answer).
    dataset = load_dataset("openai/gsm8k", "main", split="train[:100]")
    # For this example, we simply use the question as prompt and a processed answer.
    def preprocess(example):
        # Use the question as prompt and extract a simple answer (this is just for illustration)
        example["prompt"] = example["question"]
        # Here we simply use the provided answer text as the gold answer.
        example["answer"] = example["answer"]
        return example
    dataset = dataset.map(preprocess)

    # Load model and tokenizer.
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    # Ensure the tokenizer has a pad token.
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(config.model_name, torch_dtype=torch.bfloat16)
    
    # Train using GRPO.
    trained_model = train_grpo(model, dataset, tokenizer, config, reward_functions)
