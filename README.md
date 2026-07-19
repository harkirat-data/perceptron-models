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

# Model 3 : Perceptron Binary Classification 

A notebook exploring the perceptron learning algorithm on a synthetic 2D binary classification dataset, comparing a from-scratch implementation against scikit-learn's Perceptron.

Dataset

Generated with sklearn.datasets.make_classification:

pythonX, y = make_classification(
    n_samples=100, n_features=2, n_informative=1, n_redundant=0,
    n_classes=2, n_clusters_per_class=1, random_state=41,
    hypercube=False, class_sep=10
)

Note: n_informative=1 means only feature 1 (x-axis) carries class signal. Feature 2 (y-axis) is noise. This is a known limitation of the current setup — see "Open Issues" below.

Contents


Manual perceptron implementation — perceptron(X, y): trains weights via the perceptron update rule (step activation, learning rate 0.1, 1000 iterations, stochastic single-sample updates).
sklearn comparison — Perceptron(fit_intercept=True, max_iter=1000, shuffle=False, random_state=42, tol=None) fit on the same data for a baseline/sanity check.
Decision boundary visualization — extracts coef_/intercept_, computes slope m = -w1/w2 and intercept c = -b/w2, plots the line over the data using plt.plot + plt.scatter with a viridis colorbar.


Known Issues / Open Questions


Manual perceptron is unused in the plotted results. The plotted decision boundary comes from sklearn's Perceptron, not the hand-written perceptron() function. If the goal is to validate the manual implementation, it needs to be called and its weights plotted separately for comparison.
Single informative feature. With n_informative=1, the dataset is effectively 1D wearing a 2D disguise. This produces a near-vertical decision boundary (large |slope|, e.g. ~-16), which is mathematically correct but can look like a bug when the y-axis autoscales to the line instead of the data. Fix: clip axes with plt.ylim()/plt.xlim() to the data's actual range before plotting the boundary.
Zero misclassification is not guaranteed. class_sep controls average cluster separation, not overlap-free separation. Before assuming the model or the plot is wrong, check:


python   print(perceptron.score(X, y))
   print((perceptron.predict(X) != y).sum(), "misclassified")

If score < 1.0, the data isn't linearly separable as generated — no line will divide the classes perfectly. Fix by regenerating data with n_informative=2 (if a genuine 2-feature decision boundary is wanted) and/or a higher class_sep.



## Libraries Used

- NumPy
- Pandas
- Matplotlib
- Seaborn
- Scikit-Learn
- mlxtend

---


