# Llama 3.2 1B - Flash Attention

This directory is a inference implementation of Llama 3.2-1B as per [Llama 3 Herd of Models](https://arxiv.org/pdf/2407.21783).

Flash Attention 2 is integrated in the forward pass of this model. It is written using triton kernels. Flash Attention + Grouped Query Attention.
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
