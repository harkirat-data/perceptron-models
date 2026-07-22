Model 3: Perceptron Binary Classification

A notebook exploring the perceptron learning algorithm on a synthetic 2D binary classification dataset, comparing a from-scratch implementation against scikit-learn's Perceptron.

Dataset

Generated with sklearn.datasets.make_classification, called as:

X, y = make_classification(n_samples=100, n_features=2, n_informative=1, n_redundant=0, n_classes=2, n_clusters_per_class=1, random_state=41, hypercube=False, class_sep=10)

Note: n_informative=1 means only feature 1 (x-axis) carries class signal. Feature 2 (y-axis) is noise. This is a known limitation of the current setup, covered further below.

Contents

Manual perceptron implementation. The function perceptron(X, y) trains weights using the perceptron update rule: step activation, learning rate 0.1, 1000 iterations, stochastic single-sample updates.

Sklearn comparison. Perceptron(fit_intercept=True, max_iter=1000, shuffle=False, random_state=42, tol=None) fit on the same data, used as a baseline and sanity check.

Decision boundary visualization. Extracts coef_ and intercept_, computes slope m = -w1/w2 and intercept c = -b/w2, then plots the line over the data using plt.plot and plt.scatter with a viridis colorbar.

Known issues and open questions

First, the manual perceptron is unused in the plotted results. The plotted decision boundary comes from sklearn's Perceptron, not the hand-written perceptron() function. If the goal is to validate the manual implementation, it needs to be called separately and its own weights plotted for comparison against sklearn's.

Second, this is effectively a single-informative-feature dataset. With n_informative=1, the data is 1D wearing a 2D disguise. This produces a near-vertical decision boundary (large slope, roughly -16), which is mathematically correct but can look like a bug when the y-axis autoscales to the line instead of the data. Fix: clip the axes with plt.ylim() and plt.xlim() to the data's actual range before plotting the boundary.

Third, zero misclassification is not guaranteed. class_sep controls average cluster separation, not overlap-free separation. Before assuming the model or plot is wrong, check the score directly: perceptron.score(X, y), and count misclassifications with (perceptron.predict(X) != y).sum(). If score is below 1.0, the data is not linearly separable as generated, and no line will divide the classes perfectly. Fix by regenerating data with n_informative=2 if a genuine two-feature decision boundary is wanted, and/or by raising class_sep.

Requirements

numpy, pandas, matplotlib, seaborn, scikit-learn

Usage

Run cells top to bottom in Jupyter. Adjust class_sep, n_informative, and random_state in the make_classification call to change separability, then re-run downstream cells to regenerate the boundary plot.
