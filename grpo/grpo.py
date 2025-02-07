import torch
import re
import tqdm
from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass
from datasets import load_dataset, Dataset
from typing import List, Optional, Tuple, Callable
from transformers import AutoTokenizer, AutoModelForCausalLM, get_cosine_schedule_with_warmup

# ------------------------------------------------------------------------
# CONFIG
# ------------------------------------------------------------------------
@dataclass 
class GRPOConfig:
    """Config for GRPO"""

    # Generation params
    num_generations: int = 64  # G in paper 
    max_length: int = 512
    temperature: float = 0.7

    # Model params
    model_name: str = "meta-llama/Llama-3.2-1B"
    output_dir: str = "outputs/Llama-1B-GRPO"
    run_name: str = "Llama-1B-GRPO" 
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    # Training params from demo
    learning_rate: float = 5e-6
    adam_beta1: float = 0.9
    adam_beta2: float = 0.99
    weight_decay: float = 0.1
    warmup_ratio: float = 0.1
    lr_scheduler_type: str = 'cosine'

    # Optional PEFT config
    use_peft: bool = False
    peft_config = None  # Can be initialized later if use_peft=True

# ------------------------------------------------------------------------
# HELPERS
# ------------------------------------------------------------------------
def maybe_apply_chat_template(example: dict, tokenizer) -> dict:
    """Apply chat template if the example is in conversational format"""
    if isinstance(example.get("prompt"), list):  # If prompt is a list of messages
        return {"prompt": tokenizer.apply_chat_template(example["prompt"])}
    return {"prompt": example["prompt"]}  # Otherwise return as is

def is_conversational(example: dict) -> bool:
    """Check if example is in conversational format"""
    return isinstance(example.get("prompt"), list)

def format_prompt(text: str) -> str:
    """Add XML tags if not present"""
    if not text.startswith("<reasoning>"):
        return f"<reasoning>\n{text}\n</reasoning>\n<answer>\n"
    return text
# ------------------------------------------------------------------------
# DATA LOADING & PREPROCESSING
# ------------------------------------------------------------------------
SYSTEM_PROMPT = """
Respond in the following format:
<reasoning>
...
</reasoning>
<answer>
...
</answer>
"""

XML_COT_FORMAT = """\
<reasoning>
{reasoning}
</reasoning>
<answer>
{answer}
</answer>
"""

def extract_xml_answer(text: str) -> str:
    return text.split("<answer>")[-1].split("</answer>")[0].strip()

def extract_hash_answer(text: str) -> str:
    if "####" not in text:
        return None
    return text.split("####")[1].strip()

def get_gsm8k_questions(split = "train") -> Dataset:
    data = load_dataset('openai/gsm8k', 'main')[split]
    data = data.map(lambda x: {
        'prompt': [
            {'role': 'system', 'content': SYSTEM_PROMPT},
            #{'role': 'user', 'content': 'What is the largest single-digit prime number?'},
            #{'role': 'assistant', 'content': XML_COT_FORMAT.format(
            #    reasoning="9 is divisble by 3 and 8 is divisible by 2, but 7 is prime.",
            #    answer="7"
            #)},
            {'role': 'user', 'content': x['question']}
        ],
        'answer': extract_hash_answer(x['answer'])
    }) 
    return data 

# dataset = get_gsm8k_questions()

# ------------------------------------------------------------------------
# Base Model & Tokenizer Setup
# ------------------------------------------------------------------------
model = AutoModelForCausalLM.from_pretrained(
    GRPOConfig.model_name,
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
    device_map=None
).to(GRPOConfig.device)

tokenizer = AutoTokenizer.from_pretrained(GRPOConfig.model_name)
tokenizer.pad_token = tokenizer.eos_token

# ------------------------------------------------------------------------
# Reward Functions
# ------------------------------------------------------------------------
def correctness_reward_func(prompts, completions, answer, **kwargs) -> list[float]:
    responses = [completion[0]['content'] for completion in completions]
    q = prompts[0][-1]['content']
    extracted_responses = [extract_xml_answer(r) for r in responses]
    print('-'*20, f"Question:\n{q}", f"\nAnswer:\n{answer[0]}", f"\nResponse:\n{responses[0]}", f"\nExtracted:\n{extracted_responses[0]}")
    return [2.0 if r == a else 0.0 for r, a in zip(extracted_responses, answer)]

def int_reward_func(completions, **kwargs) -> list[float]:
    responses = [completion[0]['content'] for completion in completions]
    extracted_responses = [extract_xml_answer(r) for r in responses]
    return [0.5 if r.isdigit() else 0.0 for r in extracted_responses]

def strict_format_reward_func(completions, **kwargs) -> list[float]:
    """Reward function that checks if the completion has a specific format."""
    pattern = r"^<reasoning>\n.*?\n</reasoning>\n<answer>\n.*?\n</answer>\n$"
    responses = [completion[0]["content"] for completion in completions]
    matches = [re.match(pattern, r) for r in responses] 
    return [0.5 if match else 0.0 for match in matches]

def soft_format_reward_func(completions, **kwargs) -> list[float]:
    """Reward function that checks if the completion has a specific format."""
    pattern = r"<reasoning>.*?</reasoning>\s*<answer>.*?</answer>"
    responses = [completion[0]["content"] for completion in completions]
    matches = [re.match(pattern, r) for r in responses] 
    return [0.5 if match else 0.0 for match in matches]

def count_xml(text) -> float:
    count = 0.0
    if text.count("<reasoning>\n") == 1:
        count += 0.125
    if text.count("\n</reasoning>\n") == 1:
        count += 0.125
    if text.count("\n<answer>\n") == 1:
        count += 0.125
        count -= len(text.split("\n</answer>\n")[-1])*0.001
    if text.count("\n</answer>") == 1:
        count += 0.125
        count -= (len(text.split("\n</answer>")[-1]) - 1)*0.001
    return count

def xmlcount_reward_func(completions, **kwargs) -> list[float]:
    contents = [completion[0]["content"] for completion in completions]
    return [count_xml(c) for c in contents]

reward_functions = [
    xmlcount_reward_func,
    soft_format_reward_func,
    strict_format_reward_func, 
    int_reward_func,
    correctness_reward_func
]

# ------------------------------------------------------------------------
# GRPO Core Functionality
# ------------------------------------------------------------------------
def sample_group(prompt: str, model: AutoModelForCausalLM, tokenizer: AutoTokenizer, config: GRPOConfig) -> List[str]:
    """Generate G completions for a single prompt"""

    # Format prompt with XML tags
    formatted_prompt = format_prompt(prompt)
    
    # Tokenize
    inputs = tokenizer(formatted_prompt, return_tensors="pt").to(config.device)
    
    # Generate G completions
    completions = []
    for _ in range(config.num_generations):
        outputs = model.generate(
            **inputs,
            max_length=config.max_length,
            temperature=config.temperature,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id\
        )
        text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        completions.append(text)
        
    return completions
    
def calculate_group_advantages(rewards: torch.Tensor, group_size: int) -> torch.Tensor:
    """Calculate advantages using group normalization from the GRPO paper."""
    # Reshape rewards to (B, G) to process each group
    grouped_rewards = rewards.view(-1, group_size)
    
    # Calculate mean and std for each group
    group_means = grouped_rewards.mean(dim=1, keepdim=True)  # (B, 1)
    group_stds = grouped_rewards.std(dim=1, keepdim=True)    # (B, 1)
    
    # Normalize each group: (r - mean)/(std + eps)
    advantages = (grouped_rewards - group_means)/(group_stds + 1e-8)
    
    # Reshape back to (B*G)
    return advantages.view(-1)

def calculate_kl_divergence(current_logits: torch.Tensor, ref_logits: torch.Tensor) -> torch.Tensor:
    """Calculate KL divergence using unbiased estimator from GRPO paper"""
    ratio = torch.exp(ref_logits - current_logits)
    kl = ratio - (ref_logits - current_logits) - 1.0
    return kl

def calculate_policy_ratio(current_logits: torch.Tensor, old_logits:torch.Tensor):
    """Calculates new vs old ratio"""
    return torch.exp(current_logits - old_logits)

def calculate_grpo_loss(
    policy_ratio: torch.Tensor,
    advantages: torch.Tensor,
    kl_div: torch.Tensor,
    beta: float = 0.04,
    epsilon: float = 0.2,
    mask: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """Calculate GRPO loss from equation (3) in paper."""
    # PPO-style clipping
    ratio_clipped = torch.clamp(policy_ratio, 1 - epsilon, 1 + epsilon)
    
    # Policy gradient loss terms
    pg_loss1 = policy_ratio * advantages
    pg_loss2 = ratio_clipped * advantages
    pg_loss = -torch.min(pg_loss1, pg_loss2)  # Negative because we want to maximize

    # Add KL penalty
    loss = pg_loss + beta * kl_div
    
    # Apply mask if provided
    if mask is not None:
        loss = loss * mask
        loss = loss.sum() / mask.sum()  # Average over non-padded tokens
    else:
        loss = loss.mean()
        
    return loss

# ------------------------------------------------------------------------
# Training Loop
# ------------------------------------------------------------------------
# Single training step implementation
def grpo_training_step(
    batch: List[str],
    current_model: AutoModelForCausalLM,
    old_model: AutoModelForCausalLM,
    ref_model: AutoModelForCausalLM,
    reward_functions: List[Callable],
    tokenizer: AutoTokenizer,
    config: GRPOConfig,
) -> Tuple[torch.Tensor, dict]:
    """
    Execute single GRPO training step.
    Returns loss and metrics dictionary.
    """
    # 1. Generate completions
    completions = sample_group(
        prompt=batch, 
        model=current_model,
        tokenizer=tokenizer,
        config=config
    )
    
    # 2. Compute rewards
    batch_rewards = torch.zeros(len(completions), len(reward_functions))
    for i, reward_func in enumerate(reward_functions):
        rewards = reward_func(batch, completions)
        batch_rewards[:, i] = torch.tensor(rewards)
    
    # Sum rewards from all functions
    rewards = batch_rewards.sum(dim=1)
    
    # 3. Calculate advantages
    advantages = calculate_group_advantages(rewards, config.num_generations)
    
    # 4. Get logits from all models
    # Prepare inputs
    input_ids = tokenizer(completions, return_tensors="pt", padding=True).to(config.device)
    attention_mask = input_ids["attention_mask"]
    
    with torch.no_grad():
        # Get old policy logits 
        old_outputs = old_model(
            input_ids=input_ids["input_ids"],
            attention_mask=attention_mask
        )
        old_logits = old_outputs.logits

        # Get reference model logits
        ref_outputs = ref_model(
            input_ids=input_ids["input_ids"],
            attention_mask=attention_mask
        )
        ref_logits = ref_outputs.logits
    
    # Get current policy logits
    current_outputs = current_model(
        input_ids=input_ids["input_ids"],
        attention_mask=attention_mask
    )
    current_logits = current_outputs.logits
    
    # 5. Calculate ratios and KL
    policy_ratio = calculate_policy_ratio(current_logits, old_logits)
    kl_div = calculate_kl_divergence(current_logits, ref_logits)
    
    # 6. Compute loss
    loss = calculate_grpo_loss(
        policy_ratio=policy_ratio,
        advantages=advantages,
        kl_div=kl_div,
        beta=config.beta,
        mask=attention_mask
    )
    
    # Collect metrics
    metrics = {
        "loss": loss.item(),
        "mean_reward": rewards.mean().item(),
        "mean_advantage": advantages.mean().item(),
        "mean_kl": kl_div.mean().item(),
        "mean_policy_ratio": policy_ratio.mean().item()
    }
    
    return loss, metrics

# Full training loop
def train_grpo(
    model: AutoModelForCausalLM,
    train_dataset: Dataset,
    tokenizer: AutoTokenizer,
    config: GRPOConfig,
    reward_functions: List[Callable],
) -> AutoModelForCausalLM:
    """
    Train model using GRPO algorithm.
    """
    # Setup models
    # Clone models for reference and old policy
    ref_model = deepcopy(model)
    old_model = deepcopy(model)
    
    # Freeze reference and old models
    for param in ref_model.parameters():
        param.requires_grad = False
    for param in old_model.parameters():
        param.requires_grad = False
        
    # Move models to device
    model = model.to(config.device)
    ref_model = ref_model.to(config.device)
    old_model = old_model.to(config.device)
    
    # Setup optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        betas=(config.adam_beta1, config.adam_beta2),
        weight_decay=config.weight_decay
    )
    
    # Setup scheduler
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(config.warmup_ratio * len(train_dataset)),
        num_training_steps=len(train_dataset)
    )
    
    # Training loop
    model.train()
    step = 0
    best_loss = float('inf')
    
    for epoch in range(config.num_epochs):
        epoch_metrics = defaultdict(list)
        
        for batch in tqdm(train_dataset, desc=f"Epoch {epoch}"):
            # Execute training step
            loss, metrics = grpo_training_step(
                batch=batch,
                current_model=model,
                old_model=old_model,
                ref_model=ref_model,
                reward_functions=reward_functions,
                tokenizer=tokenizer,
                config=config
            )
            
            # Optimization
            loss.backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            
            # Update old model periodically
            if step % config.policy_update_steps == 0:
                old_model.load_state_dict(model.state_dict())
            
            # Log metrics
            for k, v in metrics.items():
                epoch_metrics[k].append(v)
            
            # Save best model
            if loss.item() < best_loss:
                best_loss = loss.item()
                torch.save(model.state_dict(), f"{config.output_dir}/best_model.pt")
            
            step += 1
            
        # Log epoch metrics
        print(f"Epoch {epoch} metrics:")
        for k, v in epoch_metrics.items():
            mean_v = sum(v) / len(v)
            print(f"{k}: {mean_v:.4f}")
            
    return model