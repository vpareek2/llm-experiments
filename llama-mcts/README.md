# Plan

Below is a step‐by‐step concrete plan that you can follow to build a toy MCTS-augmented GRPO system on a setup roughly equivalent to 2×RTX 4090s. This plan covers environment setup, baseline establishment, MCTS module development, integration into training/inference, and evaluation. You can later scale up the experiment if the initial results look promising.

---

## **Step 1: Environment & Setup**

1. **Hardware & Software:**
   - **Hardware:**  
     Use your 2×RTX 4090 setup. Initially, you can run the main training loop on one GPU and reserve the second for parallelizing MCTS rollouts (or use both if you implement multi-GPU data parallelism).
   - **Software:**  
     - Python 3.8+  
     - PyTorch (latest stable version with CUDA support)  
     - Hugging Face Transformers  
     - Datasets library  
     - TRL (for GRPOTrainer)  
     - wandb (for logging)  
     - Other dependencies (e.g., regex, peft)

2. **Virtual Environment:**  
   Create and activate a Python virtual environment, and install the dependencies using pip.

3. **Repository Setup:**  
   Organize your project with directories such as:
   - `src/` for source code
   - `configs/` for experiment configurations
   - `logs/` for logging outputs
   - `scripts/` for training/inference launchers

---

## **Step 2: Establish a Baseline**

1. **Baseline GRPO Training (Without MCTS):**
   - Use your provided `train_grpo.py` script to run a baseline experiment on GSM8K with your chosen model (e.g., Qwen 1.5B or LLaMA 1B).
   - Verify that the system can load data, initialize the model, and complete a short training run.
   - Monitor runtime per iteration and ensure that your reward functions (format, correctness, etc.) are working as expected.

2. **Logging & Evaluation:**
   - Use wandb (or a similar tool) to log metrics.
   - Save checkpoints and sample outputs to later compare against MCTS-augmented runs.

---

## **Step 3: Develop a Toy MCTS Module**

1. **Design Considerations:**
   - **Scope:**  
     Start by planning a shallow tree search—e.g., 2–3 tokens ahead at a time.
   - **Simulations:**  
     Run a limited number of rollouts (e.g., 3–5 simulations per decision point) to keep compute manageable.
   - **Reward Evaluation:**  
     Leverage your existing reward functions (e.g., `xmlcount_reward_func`, `strict_format_reward_func`, etc.) to score completions.

2. **MCTS Skeleton:**
   - Create a new module, for example, `mcts.py`.
   - Define a function `mcts_rollout(prompt, model, tokenizer, n_simulations, rollout_depth, reward_func)` that:
     - Takes the current prompt (or state) and generates `n_simulations` completions for `rollout_depth` tokens.
     - Uses your reward functions to evaluate each rollout.
     - Returns the “best” next token (or tokens) based on the highest cumulative reward.
   
3. **Example Code Snippet:**

   ```python
   # mcts.py
   import torch
   from transformers import AutoTokenizer, AutoModelForCausalLM

   def generate_completion(prompt, model, tokenizer, max_length, **generation_kwargs):
       inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
       outputs = model.generate(
           **inputs,
           max_new_tokens=max_length,
           **generation_kwargs
       )
       return tokenizer.decode(outputs[0], skip_special_tokens=True)

   def mcts_rollout(prompt, model, tokenizer, n_simulations=3, rollout_depth=3, reward_func=None):
       """
       For a given prompt, simulate several completions and choose the best next token(s).
       """
       candidates = []
       rewards = []
       for _ in range(n_simulations):
           # Generate a short rollout
           rollout = generate_completion(prompt, model, tokenizer, max_length=rollout_depth, do_sample=True)
           # Concatenate prompt and rollout
           candidate_output = prompt + rollout
           candidates.append(candidate_output)
           # Evaluate using reward function (assume reward_func takes a list of outputs)
           if reward_func:
               # Here we call the reward function on a list with a single candidate
               candidate_reward = reward_func([{'content': candidate_output}])[0]
           else:
               candidate_reward = 0.0
           rewards.append(candidate_reward)
       
       # Select the candidate with the highest reward
       best_idx = rewards.index(max(rewards))
       best_candidate = candidates[best_idx]
       # Extract the next token(s) generated beyond the prompt (here, all of the rollout)
       next_tokens = best_candidate[len(prompt):]
       return next_tokens, rewards
   ```

   > **Note:** This is a simplified version. In practice, you might need to handle tree data structures, backpropagation of rewards, and more sophisticated credit assignment. For a toy experiment, using this “one-step lookahead” can be a good start.

---

## **Step 4: Integrate MCTS into Your Pipeline**

1. **Integration Points:**
   - **Inference-Time MCTS:**  
     Begin by using MCTS during inference. For a given question from GSM8K, use your baseline GRPO model to generate an initial chain-of-thought, then for selected decision points, replace or augment the token generation with an MCTS rollout.
   - **Training-Time (Optional):**  
     Once inference-time integration is stable, you might experiment with integrating MCTS into GRPO training. This is more challenging since it requires modifying the trainer to perform additional rollouts per training step.

2. **Modifying the Training/Generation Loop:**
   - In your generation loop (whether in training or evaluation), add an option (e.g., via a flag) to call `mcts_rollout` instead of a plain forward pass for certain tokens or segments.
   - For example, you might decide to use MCTS for every 5th token or at points where the model’s confidence is low (if you can measure that).

3. **Testing the Integration:**
   - Write a small script or notebook cell that feeds a sample prompt to the integrated system.
   - Print out the MCTS rollouts and observe how the “best” rollout compares with a normal greedy or sampled generation.
   - Adjust parameters (number of simulations, rollout depth) until you see a reasonable improvement in the generated chain-of-thought or format adherence.

---

## **Step 5: Experiment Design & Execution**

1. **Define Experiments:**
   - **Baseline vs. MCTS-Inference:**  
     Compare outputs (e.g., chain-of-thought clarity, correctness, format) from the baseline GRPO system and the MCTS-augmented version.
   - **Ablation Studies:**  
     Vary the number of simulations and rollout depth to see how these affect performance and latency.
   - **Metrics:**  
     - GSM8K accuracy (using your correctness reward function).
     - Formatting scores (via your strict/soft format reward functions).
     - Inference time per token/answer.

2. **Compute Considerations:**
   - **Inference Latency:**  
     With 3–5 rollouts per decision, each token may take 3–5× the normal forward pass time. Given each pass is ~10–20ms on a 4090, this should remain manageable for a toy experiment.
   - **Batching:**  
     Use your 2 GPUs to parallelize rollouts where possible. For instance, one GPU could handle the baseline generation while the other handles the multiple rollouts for MCTS.

3. **Run the Experiments:**
   - Begin with a small set of GSM8K examples.
   - Log the results and compare the generated answers, rewards, and overall runtime.
   - Adjust your MCTS parameters based on early findings.

---

## **Step 6: Evaluation & Analysis**

1. **Qualitative Analysis:**
   - Compare chain-of-thought outputs from the baseline and MCTS-augmented systems.
   - Look for improvements in reasoning, format adherence, and correctness.

2. **Quantitative Analysis:**
   - Use your reward functions to score outputs.
   - Track metrics like average reward per example, percentage of correctly formatted answers, and GSM8K accuracy.

3. **Resource Monitoring:**
   - Keep an eye on GPU memory usage and computation time.
   - Determine if further batching or parameter tuning is necessary.

---

## **Step 7: Iteration & Scaling Up**

1. **Iteration:**
   - Based on initial results, iterate on the MCTS module:
     - Tweak the simulation count, rollout depth, and reward weighting.
     - Possibly add more sophisticated tree search features if needed (e.g., incorporating UCB scores for exploration vs. exploitation).
   
2. **Scaling Up:**
   - Once the toy experiment is working and showing promising improvements, consider scaling up:
     - Increase the model size or number of training epochs.
     - Move to full training-time integration of MCTS if inference-time gains are significant.
     - Use distributed training across both GPUs if the code and library support it.

---

## **Timeline & Milestones**

1. **Week 1:**  
   - Environment setup and baseline GRPO training.  
   - Validate data loading, model initialization, and reward functions.

2. **Week 2:**  
   - Develop and test the toy MCTS module.  
   - Run initial inference experiments with MCTS rollouts.

3. **Week 3:**  
   - Integrate MCTS into the generation loop.  
   - Run small-scale experiments on a subset of GSM8K.

4. **Week 4:**  
   - Analyze results, adjust parameters, and perform ablation studies.
   - Document changes and prepare for potential scaling.

5. **Beyond Week 4:**  
   - Decide whether to integrate MCTS into the training loop or scale up the experiments to larger models/datasets based on toy experiment performance.

---

## **Additional Tips**

- **Logging & Debugging:**  
  Use extensive logging (via wandb or local logging) to capture not only final rewards but also intermediate states of the tree search. This will help diagnose if the MCTS module is guiding the generation in the right direction.

- **Fallbacks:**  
  Maintain a fallback option to revert to the baseline generation if MCTS integration leads to instability. You might include a probability to use plain sampling during training/inference.

- **Community & Resources:**  
  Look for similar projects or open-source examples where MCTS was applied in NLP settings. They might offer insights on efficient batching and integration patterns.

---

Following this concrete plan should provide you with a manageable pathway to integrate MCTS into your GRPO training framework on a toy setup using 2×RTX 4090 GPUs. Once the toy experiments show promising results, you’ll be in a good position to scale up further.

If you need further clarifications or more detailed code examples for any specific step, let me know!