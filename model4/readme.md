# Manual Perceptron Implementation — model4

A from-scratch perceptron implementation trained on a synthetic 2D binary classification dataset, with the decision boundary extracted and plotted directly from the learned weights.

## Dataset

```python
X, y = make_classification(
    n_samples=100, n_features=2, n_informative=1, n_redundant=0,
    n_classes=2, n_clusters_per_class=1, random_state=41,
    hypercube=False, class_sep=15
)
```

`n_informative=1` — only feature 1 carries class signal; feature 2 is noise. Expect a near-vertical decision boundary once training is correct, since the model has little reason to weight feature 2.

## Perceptron

```python
def perceptron(X, y):
    w1 = w2 = b = 1
    lr = 0.1

    for j in range(1000):
        for i in range(X.shape[0]):
            z = w1*X[i][0] + w2*X[i][1] + b
            if y[i]*z <= 0:
                w1 = w1 + lr*y[i]*X[i][0]
                w2 = w2 + lr*y[i]*X[i][1]
                b = b + lr*y[i]
    return w1, w2, b
```

- Weights and bias initialized to 1 (not zero — deliberate choice, worth revisiting if convergence behavior looks off).
- Misclassification check is `y[i]*z <= 0` (not `z*X[i]`, which is dimensionally meaningless and throws an ambiguous-truth-value error on arrays).
- `<=` rather than `<` so points exactly on the boundary still trigger an update.

Decision boundary is recovered from the weights as a line:
```python
m = -(w1/w2)
c = -(b/w2)
```
and plotted with `plt.plot(x_input, m*x_input + c, ...)`, `x_input` swept with `np.linspace`.

## Bugs found and fixed during development

1. **Wrong misclassification condition.** Original code used `if z*X[i] < 0`, multiplying a scalar by the feature vector — dimensionally wrong and throws `ValueError: truth value of an array with more than one element is ambiguous`. Corrected to `if y[i]*z <= 0`, the actual perceptron misclassification test.

2. **`return` indented inside the inner loop.** This caused the function to evaluate a single data point (`i=0`) and return immediately, regardless of the outer 1000-epoch loop — so training never actually ran on more than one sample. Symptom: `w1, w2, b` came back unchanged from their initial values (`1, 1, 1`), and the plotted "decision boundary" was just the untrained initialization, not a fitted line. Fixed by dedenting `return` to sit outside both `for` loops.

## Verifying a genuine fit

Before trusting the boundary visually, confirm training actually happened and check separability:
```python
w1, w2, b = perceptron(X, y)
print(w1, w2, b)   # should NOT be 1, 1, 1

z = w1*X[:,0] + w2*X[:,1] + b
preds = np.where(z > 0, 1, -1)   # match to however y is encoded
misclassified = (preds != y).sum()
print(misclassified)
```
If misclassified count isn't 0, the classes may genuinely overlap for this `random_state`/`class_sep` — that's a data separability issue, not necessarily a code bug. Increase `class_sep` or check the scatter plot before assuming the model is broken.


