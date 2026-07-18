This repository demonstrates the implementation of the Perceptron algorithm on both a simple two-feature dataset and a higher-dimensional real-world dataset.

The goal of this repository is not just to train a classifier, but to build an intuitive understanding of how a Perceptron learns, how decision boundaries are formed, and what happens when the dimensionality of the data increases.

---

## Repository Structure

```
.
├── model1/
│   ├── placement_dataset-model.ipynb
│   └── placement_dataset.csv
│
├── model2/
│   ├── Diabetes_dataset-model.ipynb
│   └── diabetes.csv
│
└── README.md
```

---

# Model 1 : Perceptron on Placement Dataset

## Dataset

Features

- CGPA
- Resume Score

Target

- Placed (0 = No, 1 = Yes)

---

## Objective

This notebook introduces the Perceptron algorithm on a simple 2-dimensional dataset to build geometric intuition.

Topics covered

- Exploratory Data Analysis
- Scatter Plot Visualization
- Feature Selection
- Perceptron Training
- Learned Weights
- Learned Bias
- Decision Boundary Visualization

---

## What I Learned

- Binary Classification
- Hyperplane in 2D
- Weight Updates
- Bias
- Linear Separability
- Decision Regions

---

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

## Libraries Used

- NumPy
- Pandas
- Matplotlib
- Seaborn
- Scikit-Learn
- mlxtend

---


