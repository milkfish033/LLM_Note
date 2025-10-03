# MLA in Deepseek v2

## MLA 技术的原理和实现

- 时间复杂度：**O(n)**  

---

### MHA (Multi-Head Attention)  
- 基本组成：**Query, Key, Value**  
- 针对 KV 提出的优化策略：**KV Cache**  

---

### 优化 KV Cache 的方法

1. **MQA (Multi-Query Attention)**  
   - 对于不同的 query，共享同一个 k, v  

2. **GQA (Group-Query Attention)**  
   - 对 query 进行分组  
   - 同一个 group 内的 query 共享同一个 k, v  

---

### 进一步优化：降维与矩阵吸收

- 将 `WKh^T` 变为 `WU KW DKV h^T`，实现降维  
- 提前计算好 `Wq^T * WUK`  
- 在 decode 时，计算量没有增加，即 **矩阵吸收**  

---

### MLA 与 RoPE 位置编码的不兼容性

- 使用 RoPE 后，无法通过恒等变换减少 KV Cache  
- 解决方案：  
  - 在 Q, K 上新加 d 个维度，单独储存位置向量  
  - 推理时，缓存 `c_tKV` 和 `K_tR`  

---

## 参考资料
- [一文了解 Deepseek 系列中的 MLA 技术](https://github.com/luhengshiwo/LLMForEverybody/blob/main/01-%E7%AC%AC%E4%B8%80%E7%AB%A0-%E9%A2%84%E8%AE%AD%E7%BB%83/%E4%B8%80%E6%96%87%E4%BA%86%E8%A7%A3Deepseek%E7%B3%BB%E5%88%97%E4%B8%AD%E7%9A%84MLA%E6%8A%80%E6%9C%AF.md)
