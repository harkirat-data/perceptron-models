# Model 2 : Perceptron on Diabetes Dataset

Unlike Model 1, this notebook uses a real-world dataset containing eight input features.

This demonstrates how Perceptron behaves in higher-dimensional feature spaces.

---

## Dataset

Features

- Pregnancies
- Glucose
- BloodPressure
- SkinThickness
- Insulin
- BMI
- DiabetesPedigreeFunction
- Age

Target

- Outcome

---

## Data Preprocessing

Performed the following preprocessing steps:

- Missing value detection
- Replaced invalid zero values using Median Imputation
- Outlier Analysis using Boxplots
- Feature Scaling using StandardScaler
- Train-Test Split

---

## Model Training

The Perceptron was trained using all **8 input features**.

The notebook includes

- Model Training
- Learned Weights
- Learned Bias
- Accuracy
- Confusion Matrix
- Classification Report

---

# PCA for Visualization

Visualizing an 8-dimensional decision boundary is impossible.

To build geometric intuition, Principal Component Analysis (PCA) was applied.

The original feature space

```
8 Dimensions
```

was reduced to

```
2 Principal Components
```

This allowed visualization of the dataset along with the Perceptron's learned decision boundary.

> **Note:** PCA was used only for visualization. The primary model was trained on the original 8-dimensional feature space.

---




