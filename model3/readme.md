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


