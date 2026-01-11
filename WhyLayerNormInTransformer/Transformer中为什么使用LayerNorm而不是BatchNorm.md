
[中文](./Transformer中为什么使用LayerNorm而不是BatchNorm.md) |
[English](./WhyLayerNormInTransformer.md)

## 归一化和标准化的区别是什么

- **归一化**：通过某种规则，把数值映射到一个“更稳定、更可比较”的尺度或分布上
- **标准化**：将数据缩放到均值为 0，方差为 1 的正态分布  

## 为什么需要归一化

- 归一化导数值范围稳定，从而使得梯度稳定
- 数值范围稳定，避免数值过大和过小，从而避免落入激活函数的饱和区间，导致梯度爆炸

![LayerNorm_vs_BatchNomr](../images/LNvsBN.png)

# Batch Normolization 
## 基本思想： 在统一特征上计算不同样本
cons:
   - 不同样本的特征长度不一样，导致均值和方差大，不适合变长数据
   - 如果batch size过小，也会导致均值方差不稳定
   - 新句子的长度如果超出训练范围，训练结果不适用，泛化能力差
   - 由于BN是在统一特征不同样本上做计算，输出依赖batch内其他样本，在transformer中，同一个batch里的句子可能完全不一样，导致训练污染，从而导致同一个token在不同的batch中attention score不一样，注意力变成**抖动的**
   - 训练时使用的是mini batch的均值和方差，但推理时使用的是全局均值和方差，导致训练和推理不一致

## 为什么在CNN或CV领域中使用BN？
- 因为在CNN中，BN的噪声被当作一种正则化，类似dropout的效果。但是在Transformer中，这种被当作一种非结构化噪声，不遵循语义
- 像素和像素之间的可比性仍然保留，而使用LN丧失了这种可比性

# LayerNorm

## 基本思想： 在统一样本上计算不同特征


特点：  
- 对batch size不敏感，由于是对统一样本不同特征做归一化，对样本长度没有要求，适用于变长数据
- 减少 BN 中带来的噪声问题， 均值和方差都来自于当前token，不考虑其他样本，更稳定
- 更具有语言意义， Transformer的hidden state本质是一个语义向量

---

## Transformer 中为什么使用 LayerNorm 而不是 BatchNorm

Transformer 使用 LayerNorm 而不是 BatchNorm，是因为 LayerNorm 不依赖 batch 统计量，适合变长序列和小 batch 场景；它对每个 token 独立归一化，确保归一化数据后本质上是可比的，与自注意力和残差结构天然匹配，并保证训练与推理阶段行为一致，而 BatchNorm 会引入跨样本干扰和不稳定性

总结：
1 NLP领域，确保归一化数据后是可比的
2 LN适用于变长数据，具有推理训练一致性，天然符合attention计算，更具有实际语言意义