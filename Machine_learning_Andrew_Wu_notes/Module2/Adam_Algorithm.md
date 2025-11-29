# 1. what is Adam Algorithm

Adam: Adaptive Moment Estimation. It has different learning rate for each paramter

For example, if there are 10 paramters from W1 to W10, there will be 11 learning rate in total : One for each Wi and one for b.

# 2. How it works

- if Wj(or b) keeps increasing in same direction : increase learning rate
- if Wj(or b) keeps oscillating : overshoot. Solved by reducing learning rate

**Using Adam can make the model more robust to choice of learning rate you make.**

**Robust**: A model continues to perform well even when input data is imperfect, noisy, or slightly different from training data.



