# Tokenizer 分词器

## 三种常见分词方式
- **Word level**  
- **Char level**  
- **Subword level**  

---

## BPE (Byte Pair Encoding)
- **应用**：GPT, BART  
- **具体方法**：  
  1. 把每个字符作为一个 token  
  2. 统计所有相邻 token 的频率  
  3. 合并频率最高的两个 token 作为新 token  
  4. 重复上述过程（不删除已有组合）  

---

## WordPiece
- **应用**：BERT  
- **具体方法**：  
  1. 把每个字符作为一个 token  
  2. 计算两个 token 的得分：  
     \[
     score = \frac{freq(pair)}{freq(first) \times freq(second)}
     \]  
     或  
     \[
     score = \log(freq(pair)) - (\log(freq(first)) + \log(freq(second)))
     \]  
  3. 合并得分最高的两个 token 作为新 token  
  4. 重复  

---

## Unigram
- **构造词汇表**：  
  - 统计每个字符的频率  
  - 计算每个 token 的概率：  
    \[
    P(token) = \frac{\#token}{\#total}
    \]  

- **例子**：  
  \[
  P(a,b,c) = P(a) \cdot P(b) \cdot P(c)
  \]  
  \[
  P(ab, c) = P(ab) \cdot P(c)
  \]  
  \[
  P(a, bc) = P(a) \cdot P(bc)
  \]  
  \[
  P(abc) = \max(P(a,b,c), P(ab,c), P(a,bc))
  \]  

- **删除 Token 过程**：  
  示例：  
"hug": ["hug"] (score 0.071428)
"pug": ["pu", "g"] (score 0.007710)
"pun": ["pu", "n"] (score 0.006168)
"bun": ["bu", "n"] (score 0.001451)
"hugs": ["hug", "s"] (score 0.001701)

初始 Loss：  
10*(−log(0.071428))

5*(−log(0.007710))

12*(−log(0.006168))

4*(−log(0.001451))

5*(−log(0.001701)) = 169.8


- 删除 token `"hug"`：  
  - 影响 `"hug"` 和 `"hugs"`  
  - 更新后：  
    ```
    "hug": ["hu", "g"] (score 0.006802)
    "hugs": ["hu", "gs"] (score 0.001701)
    ```
  - Loss 变化：  
    ```
    hug: -10*(−log(0.071428)) + 10*(−log(0.006802)) = 23.5
    hugs: 0
    总变化 = 23.5
    ```

- **迭代所有 token，删除影响最小的一个 → 重复**  

---

## SentencePiece
- 基于 BPE  
- 把句子切分成更小的片段 → 合并  

---

## Unicode
- 适用于中文等其他语言的分词  

---

## BBPE (Byte-Level BPE)
- **核心思想**：将文本中的 **字节对（UTF-8 编码）** 合并，直到达到词汇表大小或无法继续合并  
- **与 BPE 的区别**：  
- BPE：基于字符 (character)  
- BBPE：基于字节 (byte)  

- **优点**：  
- **跨语言通用性**：基于字节，更易跨语言和不同文字体系迁移  
- **减少词汇表大小**：合并字节对生成更通用的子词单元  
- **处理罕见字符 (OOV) 问题**：不为罕见字符单独建词，而是当作字节序列处理  
