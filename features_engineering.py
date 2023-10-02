import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sys import argv
script, DEGREE = argv

matplotlib.use('TkAgg')


def plot_training_data():
    plt.figure(figsize=(7, 5))
    ax = sns.scatterplot(x='test_1', y='test_2', hue='label', data=data, style='label', s=80)
    ax.legend(['accepted', 'rejected'])
    plt.title('Scatter plot of training data')
    plt.show()

def map_feature(x1, x2, degree):
    # reshaping is done so code will be consistent with other part of the code
    x1 = np.array(x1).reshape(-1, 1)
    x2 = np.array(x2).reshape(-1, 1)

    out = np.ones((x1.shape[0], 1))
    for i in range(1, int(degree) + 1):
        for j in range(0, i + 1):
            p = (x1 ** (i - j)) * (x2 ** j)
            out = np.append(out, p, axis=1)
    # TODO
    return out


def plot_decision_boundary(coefficients=None):
    fig, axs = plt.subplots(nrows=1, ncols=1)
    u = np.linspace(-1, 1.5, 50)
    v = np.linspace(-1, 1.5, 50)

    z = np.zeros((u.shape[0], v.shape[0]))
    for i in range(len(u)):
        for j in range(len(v)):
            z[i, j] = map_feature(u[i], v[j], DEGREE).dot(coefficients)

    sns.scatterplot(x='test_1', y='test_2', hue='label', data=data, style='label', s=80, ax=axs)

    axs.contour(u, v, z.T, levels=[0], colors='green')
    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    data = pd.read_csv('data/non_linear.txt', sep=',', header=None)
    data.columns = ['test_1', 'test_2', 'label']

    plot_training_data()  # plot data to see initial points, they are not linearly separable in 2-D

    # creating new features and training logistic regression
    X = map_feature(data.test_1.values, data.test_2.values, DEGREE)

    log_reg = LogisticRegression(max_iter=400)
    log_reg.fit(X[:, 1:], data.label.values)

    # concatenating coefficient of linear regression for plotting decision boundary
    log_reg.intercept_ = np.expand_dims(log_reg.intercept_, axis=1)
    coefficients = np.squeeze(np.concatenate([log_reg.intercept_, log_reg.coef_], axis=1))

    plot_decision_boundary(coefficients=coefficients)
