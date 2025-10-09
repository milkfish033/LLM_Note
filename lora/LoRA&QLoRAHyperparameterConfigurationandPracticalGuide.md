
[中文](./lora微调的参数怎么选择，lora和Qlora的区别.md) |
[English](./LoRA&QLoRAHyperparameterConfigurationandPracticalGuide.md)
# LoRA & QLoRA Hyperparameter Configuration and Practical Guide

## I. Overview of LoRA and QLoRA
**LoRA (Low-Rank Adaptation)** and **QLoRA** are parameter-efficient fine-tuning (PEFT) techniques designed to reduce the computational cost of training large models while maintaining performance.

- **LoRA**: Inserts low-rank adapter matrices (A and B) into Transformer layers and only trains these adapters, significantly lowering the training cost.  
- **QLoRA**: Builds on LoRA by introducing **4-bit quantization**, which compresses frozen pre-trained weights and dynamically dequantizes them during computation, further reducing GPU memory usage and enabling fine-tuning of large models even on consumer-grade GPUs.

| Feature         | LoRA                              | QLoRA                                  |
|-----------------|------------------------------------|-----------------------------------------|
| Core Technique  | Low-rank adapters                  | LoRA + 4-bit quantization              |
| Memory Usage    | Lower than full fine-tuning, but higher than QLoRA | Lowest; enables training larger models on a single GPU |
| Training Speed  | Faster, no quantization overhead   | Slightly slower due to dynamic dequantization |
| Accuracy        | Often slightly higher than QLoRA   | May have minor accuracy drop, usually negligible |
| Best for        | Moderate compute resources & focus on performance | Resource-constrained setups, low GPU memory, or ultra-large models |

---

## II. LoRA Fine-Tuning Workflow
1. Freeze the original pre-trained model weights.  
2. Add **two small matrices (A and B)** alongside Transformer weight matrices to learn incremental updates ΔW.  
3. Train only the parameters of A and B.  
4. During inference, merge LoRA weights into the original weights \( W₀ + BA \), achieving the same inference speed as full fine-tuning.

---

## III. QLoRA Fine-Tuning Workflow
1. Quantize the model from 16-bit floating-point to **4-bit NF4** format.  
2. Freeze the quantized weights.  
3. Insert LoRA adapters and train with mixed precision:  
   - Dynamically dequantize weights to 16-bit for computation during training.  
   - Update only the LoRA adapter parameters.  
4. Use **Double Quantization** to further reduce GPU memory consumption.

---

## IV. Key Hyperparameters

### 1. Rank (r)
- **Purpose**: Controls the dimension of the low-rank matrices, i.e., the “bandwidth” for adapting the model to the new task.  
- Typical ranges:
  - Simple tasks: 4–8  
  - Medium tasks: 16–32  
  - Complex reasoning: 64–128  
  - Language transfer: 128–256  
- **Tuning strategy**: Start small and increase gradually to find the performance “elbow point.”

---

### 2. Scaling Factor (lora_alpha)
- **Purpose**: Determines how much influence the adapter has over the base model.  
- Common default: `alpha = 2 × r`  
- Tuning tips:
  - If training is unstable → reduce `alpha`  
  - If the model underfits → increase `alpha`

---

### 3. Target Modules (target_modules)
- **Purpose**: Specifies which layers get LoRA adapters.  
- Common choices:
  - Minimal: `["q_proj", "v_proj"]`  
  - Balanced: `["q_proj", "k_proj", "v_proj", "o_proj"]`  
  - Maximum (including FFN layers): add `["gate_proj", "up_proj", "down_proj"]`

---

### 4. Dropout (lora_dropout)
- **Purpose**: Prevents overfitting.  
- Typical range: 0.0–0.2 (commonly 0.05–0.1)  
- Low-data scenarios → higher dropout  
- Large-data scenarios → lower dropout

---

### 5. Bias
- Options:
  - `"none"`: do not train bias (most common)  
  - `"lora_only"`: train only the adapter bias  
  - `"all"`: train all biases (adds parameters)

---

### 6. Learning Rate (learning_rate)
- Typical ranges:
  - LoRA: 1e-4 ~ 5e-4  
  - QLoRA: 5e-5 ~ 2e-4 (slightly lower)

---

## V. Tuning Strategy
1. **Quick Scan**: Start with short runs at rank values [8, 16, 32, 64].  
2. **Analyze Results**: Look for stable performance and diminishing returns.  
3. **Fine-Grained Tuning**: Experiment around the best rank values.  
4. **Full Training**: Validate the top two rank settings with longer runs.

---

## VI. Common Issues & Solutions
1. **Training instability** → lower learning rate or `alpha`, apply gradient clipping.  
2. **Overfitting** → increase dropout, lower rank, or use early stopping.  
3. **Underfitting** → increase rank or expand target modules.  
4. **Out-of-memory** → switch to QLoRA or reduce batch size.

---

## References
1. [LoRA/QLoRA Hyperparameter Guide – Optimal Strategies and Configurations for Best Practices (Tableau) – Zhihu](https://zhuanlan.zhihu.com/p/1938278622218650334)  
2. [Detailed Explanation of LoRA Fine-Tuning Parameters – Zhihu](https://zhuanlan.zhihu.com/p/1953953705192821962)
