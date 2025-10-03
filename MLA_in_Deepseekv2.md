MLA in Deepseek v2 O(n)
https://github.com/luhengshiwo/LLMForEverybody/blob/main/01-%E7%AC%AC%E4%B8%80%E7%AB%A0-%E9%A2%84%E8%AE%AD%E7%BB%83/%E4%B8%80%E6%96%87%E4%BA%86%E8%A7%A3Deepseek%E7%B3%BB%E5%88%97%E4%B8%AD%E7%9A%84MLA%E6%8A%80%E6%9C%AF.md

MHA(multi-head-attention) : query , key, value 
对于kv提出缓存的优化策略：Kv Cache 

优化Kv Cache：
MQA：Multi-query-attention:对于不同的query，共享同一个k，v
GQA：Group-query-attentiomn: 对query进行分组，同一个group内的query共享同一个k，v

将WKht 变为 WUKWDKVht，进行降纬
同时，提前计算好WtqT∗WtUK， 这样在decode的时候，计算量没有增加，i.e. 矩阵吸收

MLA与RoPE位置编码不兼容（用了RoPE之后无法用恒等变化减少KV Cache）：
在Q，K上新加d个纬度，单独储存位置向量， 在推理时，缓存 ctKV 和 KtR
