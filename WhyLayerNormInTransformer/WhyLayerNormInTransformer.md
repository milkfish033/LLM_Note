
[中文](./Transformer中为什么使用LayerNorm而不是BatchNorm.md) |
[English](./WhyLayerNormInTransformer.md)

# LayerNorm

## Core Idea
- Perform normalization for each layer of a neural network  
- Compute the **mean** and **variance** of activations within a single sample at the current layer  
- Use these statistics to **normalize** the activations  

**Key Features:**  
- Independent of batch size  
- Reduces noise introduced by Batch Normalization (BN)  
- Allows training with smaller batch sizes  

---

## Why Transformer Uses LayerNorm Instead of BatchNorm

1. **To Solve the Noise Problem Introduced by BN**  
   - The noise issue in BN:  
     - During training, BN estimates mean and variance using mini-batches, which introduces estimation errors  
     - During inference, BN switches to global moving-average statistics, leading to inconsistency  

2. **To Ensure the Data Being Normalized Is Comparable in Nature**  

3. **BN’s Normalization Mechanism**  
   - BN focuses on one column of data within a batch (i.e., the same feature across different samples)  
   - Since these features are statistically related, it makes sense to normalize them together  

4. **Scenarios Where LN Excels**  
   - LN is better suited for situations where samples cannot be directly compared  
   - In the Transformer’s **self-attention mechanism**:  
     - Each position represents different features  
     - LN normalizes all features **within each individual sample**
