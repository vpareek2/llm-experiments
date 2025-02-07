import torch
import torch.nn as nn
import torch.nn.functional as F
import re

from dataclasses import dataclass
from datasets import load_dataset, Dataset
from typing import List, Optional, Tuple
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig

# ------------------------------------------------------------------------
# CONFIG
# ------------------------------------------------------------------------
@dataclass
class GRPOConfig:
    """Config for GRPO"""

    num_generations: int = 64 # G in paper
    max_length: int = 512
    temperature: float = 0.7

    model_name: str = "meta-llama/Llama-3.2-1B"
    output_dir: str = "outputs/Llama-1B-GRPO"
    run_name: str = "Llama-1B-GRPO"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

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

# ------------------------------------------------------------------------
# GRPO Core Functionality
# ------------------------------------------------------------------------
def sample_group(prompt: str, model: AutoModelForCausalLM, tokenizer: AutoTokenizer, config: GRPOConfig) -> List[str]:
    """
    Generate G completions for a single prompt
    """

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
            pad_token_id=tokenizer.pad_token_id
        )
        text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        completions.append(text)
        
    return completions
    
def calculate_group_advantages(rewards: List[float], group_size: int):
    """Calculates advantages as per DeepSeekMath paper.
    Takes rewards for a group of outputs 
    Calculates mean and std
    Returns normalized advantages using formula: advantages = (rewards - mean(rewards)) / std(rewards)
    """
    pass

def calculate_kl_divergence(current_logits, ref_logits):
    """Implements KL divergence estimator from paper"""
    pass

def calculate_policy_ratio(current_logits, old_logits):
    """Calculates new vs old ratio"""

def calculate_grpo_loss(policy_ratios, advantages, kl_divergence):
    """Calculates GRPO loss as per paper"""
    pass

# ------------------------------------------------------------------------
# Training Loop
# ------------------------------------------------------------------------
def grpo_training_step(model, ref_model, old_model, batch, rewards):
    """Implements a single GRPO training step
    Forward passes through current, reference and old policy models
    Calculates advantages, ratios, KL
    Compute loss and do optimization step
    """
    pass
# ------------------------------------------------------------------------
# Evaluation
# ------------------------------------------------------------------------