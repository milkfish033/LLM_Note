
[中文](./Tokenlizer_cn.md) |
[English](./Tokenlizer_en.md)

# Tokenizer

## Three Common Tokenization Methods
- **Word level**  
- **Character level**  
- **Subword level**  

---

## BPE (Byte Pair Encoding)
- **Applications**: GPT, BART  
- **Procedure**:  
  1. Treat each character as a token.  
  2. Count the frequency of all adjacent token pairs.  
  3. Merge the most frequent token pair into a new token.  
  4. Repeat the process (do not remove existing merges).  

---

## WordPiece
- **Applications**: BERT  
- **Procedure**:  
  1. Treat each character as a token.  
  2. Compute the score for each token pair:  
     \[
     score = \frac{freq(pair)}{freq(first) \times freq(second)}
     \]  
     or  
     \[
     score = \log(freq(pair)) - (\log(freq(first)) + \log(freq(second)))
     \]  
  3. Merge the pair with the highest score into a new token.  
  4. Repeat.  

---

## Unigram
- **Building the Vocabulary**:  
  - Count the frequency of each character.  
  - Compute the probability of each token:  
    \[
    P(token) = \frac{\#token}{\#total}
    \]

- **Example**:  
  \[
  P(a,b,c) = P(a) \cdot P(b) \cdot P(c)
  \]  
  \[
  P(ab,c) = P(ab) \cdot P(c)
  \]  
  \[
  P(a,bc) = P(a) \cdot P(bc)
  \]  
  \[
  P(abc) = \max(P(a,b,c), P(ab,c), P(a,bc))
  \]

- **Token Deletion Process**:  
    Example:  
    "hug": ["hug"] (score 0.071428)
    "pug": ["pu", "g"] (score 0.007710)
    "pun": ["pu", "n"] (score 0.006168)
    "bun": ["bu", "n"] (score 0.001451)
    "hugs": ["hug", "s"] (score 0.001701)

Initial Loss:  
    10 * (−log(0.071428))
    5 * (−log(0.007710))
    12 * (−log(0.006168))
    4 * (−log(0.001451))
    5 * (−log(0.001701)) = 169.8


- **Delete token `"hug"`**:  
- Affects `"hug"` and `"hugs"`  
- After deletion:  
  ```
  "hug": ["hu", "g"] (score 0.006802)
  "hugs": ["hu", "gs"] (score 0.001701)
  ```
- Loss change:  
  ```
  hug: -10 * (−log(0.071428)) + 10 * (−log(0.006802)) = 23.5
  hugs: 0
  Total change = 23.5
  ```

- **Iterate over all tokens → delete the one with the smallest impact → repeat**  

---

## SentencePiece
- Based on BPE  
- Splits sentences into smaller pieces → merges them into tokens.  

---

## Unicode
- Suitable for languages like Chinese and others that are not space-delimited.  

---

## BBPE (Byte-Level BPE)
- **Core Idea**: Merges **byte pairs (UTF-8 encoded)** until the desired vocabulary size is reached or no further merges are possible.  
- **Difference from BPE**:  
- BPE: based on characters  
- BBPE: based on bytes  

- **Advantages**:  
- **Cross-lingual universality**: Works across different writing systems since it operates on bytes.  
- **Smaller vocabulary size**: Combines byte pairs into more general subword units.  
- **Handles rare characters (OOV)**: Treats rare characters as byte sequences instead of assigning unique tokens.

