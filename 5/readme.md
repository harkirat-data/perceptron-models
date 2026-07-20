# Problem 5 : Perceptron Limitations (XOR Problem)

## Overview

This notebook explores the geometric limitations of a single-layer Perceptron when applied to non-linearly separable problems such as the XOR logic gate.

---

## Datasets Explored

Three synthetic binary logic truth tables:
- **AND Gate**: Linearly separable (output 1 only when both inputs are 1)
- **OR Gate**: Linearly separable (output 1 if either input is 1)
- **XOR Gate**: Non-linearly separable (output 1 if exactly one input is 1)

---

## Observations & Decision Regions

- **Linear Separability**: The Perceptron successfully converges on the AND and OR gates because a single linear decision hyperplane ($w_1 x_1 + w_2 x_2 + b = 0$) can completely divide the classes.
- **The XOR Failure**: For XOR, points `(0, 1)` and `(1, 0)` belong to class 1, while `(0, 0)` and `(1, 1)` belong to class 0. No single straight line can separate these two clusters.
- **Key Takeaway**: Demonstrates the mathematical necessity of Multi-Layer Perceptrons (MLPs) with hidden layers and non-linear activation functions to solve non-linear patterns.
