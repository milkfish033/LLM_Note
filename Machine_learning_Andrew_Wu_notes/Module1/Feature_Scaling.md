# **1. What is Feature Scaling**

Feature scaling is the process of transforming numerical features so they exist on a similar scale.  
This is important because many machine learning algorithms behave poorly when features have very different magnitudes.

For example, suppose we have two variables:

- \( x = 3000 \)
- \( y = 0.01 \)

In this situation, the model may treat **\(x\)** as far more important simply because its value is larger — not because it truly has more predictive power. This imbalance can lead to:

- Slow or unstable gradient descent  
- Poor convergence  
- Distorted distance metrics (important for KNN, K-means, etc.)  
- Model coefficients dominated by large-scale features (e.g., linear/logistic regression)

To solve this, we apply **feature scaling**, which ensures all features contribute more fairly.

---

## **Common Feature Scaling Methods**

### **1. Standardization (Z-score Normalization)**  
Transforms data to have mean 0 and standard deviation 1:

$$
x' = \frac{x - \mu}{\sigma}
$$

---

### **2. Min–Max Scaling**  
Maps each feature to the range [0, 1]:


$$x' = \frac{x - x_{\min}}{x_{\max} - x_{\min}}$$


---

### **3. Robust Scaling**  
Uses median and IQR (interquartile range):  
Useful when the data contains outliers.


