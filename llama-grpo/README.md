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

Example: 
Question: The number of dishes Sandrine washed was 10 more than the number of bananas Charles cooked. Also, the number of bananas Charles cooked was 3 times the number of pears he picked. If Charles picked 50 pears, how many dishes did Sandrine wash? 
Answer:
160 

Response:
<reasoning>
Charles picked 50 pears, so the number of bananas he cooked is 3 * 50 = 150. Sandrine washed 10 more dishes than the number of bananas Charles cooked, so she washed 150 + 10 = 160 dishes.
</reasoning>
<answer>
160
</answer>
 
Extracted:
160


Question:
After being contracted to build 4000 bollards on each side of a road, a company was only able to install 3/4 of the total number of bollards required on all sides of the road. How many more bollards are they required to install on all sides of the remaining part of the road? 
Answer:
2000 
Response:
<reasoning>
The company initially needed to build a total of 4000 bollards per side, so the total number of bollards needed for both sides of the road would be 4000 x 2 = 8000 bollards. The company installed 3/4 of the required total, so they installed 8000 x 3/4 = 6000 bollards. Therefore, they need to install 8000 - 6000 = 2000 more bollards.
</reasoning>
<answer>
2000
</answer>
 
Extracted:
2000