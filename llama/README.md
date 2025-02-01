# Base LLM Implementation

This directory is a inference implementation of Llama 3.2-1B. It is meant to be used as a base LLM for research experiments.

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

## Credit

This implementation is inspired by the [xjdr-alt/entropix repository](https://github.com/xjdr-alt/entropix/tree/main). I like the programming style.