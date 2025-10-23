[中文](./DecisionTreeCn.md) |
[English](./DecisionTreeEn.md)
# Decision Trees — Overfitting, Underfitting & Gini Index

---

## Overfitting vs Underfitting

| Term | Description | Cause | Effect |
|------|--------------|--------|--------|
| **Overfitting** | Model fits the training data too closely, capturing noise or random fluctuations. | Tree too deep, too many splits, or insufficient pruning. | Low training error but **high test error**. |
| **Underfitting** | Model is too simple to capture the underlying trend in data. | Shallow tree, too few splits, or excessive early stopping. | High error on both training and test sets. |

**Goal:** Find the balance where the tree generalizes well to unseen data.

---

## Decision Tree and Overfitting

- Decision trees tend to overfit if allowed to grow without constraints.  
- They perform **binary** or **multi-way** splits on attributes to minimize impurity.  
- Always prefer the **most informative** split based on a purity measure (e.g., Gini Index or Entropy).

**Common ways to prevent overfitting:**
1. **Early termination:**  
   Stop growing before the tree perfectly classifies the data.  
   - Limit maximum depth.  
   - Stop if the node size is below a threshold.  
   - Stop if Gini improvement is insignificant.
2. **Pruning:**  
   Grow the full tree first, then remove weak branches afterward.

---

## Gini Index

The **Gini Index** measures the impurity of a node — how mixed the classes are.

### Formula

\[
\text{Gini}(t) = 1 - \sum_j [p(j|t)]^2
\]

where \( p(j|t) \) is the relative frequency of class \( j \) at node \( t \).

---

### Examples

- **Perfect node (pure class):**  
  \( \text{Gini} = 0 \)

- **Worst node (two equally likely classes):**  
  \( \text{Gini} = 1 - (1/2)^2 - (1/2)^2 = 0.5 \)

---

### Gini of a Split

\[
\text{Gini}_{split} = \sum_t \frac{n_t}{n} \cdot \text{Gini}(t)
\]

where:
- \( n_t \): number of samples in child node \( t \)
- \( n \): total samples before the split

The **best split** is the one that minimizes \( \text{Gini}_{split} \) (or equivalently, maximizes the Gini Gain).

---

### Example from Lecture

Given:
- Parent Gini = 0.49  
- Split 1: \( \text{Gini}_{split} = 0.23 \)
- Split 2: \( \text{Gini}_{split} = 0.47 \)

Then:
\[
\text{Gain}_1 = 0.49 - 0.23 = 0.26 \quad \text{(better split)}
\]
\[
\text{Gain}_2 = 0.49 - 0.47 = 0.02
\]

Thus, choose **Split 1** since it yields the higher purity improvement.

---

## Summary

- **Overfitting** occurs when the decision tree becomes too complex.  
- **GINI index** provides a measure of node impurity — lower values indicate purer nodes.  
- Use **pruning** and **stopping criteria** to control overfitting.  
- **Underfitting** happens when the model is too simple to capture patterns in the data.

---

**Reference:** Boston University CS 506 — Decision Trees Lecture (Lance Galletti)
