# Base LLM Implementation

This directory is a inference implementation of Llama 3.2-1B as per [Llama 3 Herd of Models](https://arxiv.org/pdf/2407.21783). It is meant to be used as a base LLM for research experiments.

I have added GRPO from the DeepSeekMath paper. I wrote GRPO from scratch in `grpo.py` and it has will brown's implementation in `grpo_trl.py`, with just calling trl for grpo.

## Setup 

Install the requirements:

```
pip install -r requirements.txt
```

Then download the weights with

```
python download.py \
    --model-id meta-llama/Llama-3.2-1B-Instruct \
    --out-dir Llama-3.2-1B-Instruct
```

## Usage

To use the `llama` model, adjust the prompt in `main.py` and then run
```
python main.py
```