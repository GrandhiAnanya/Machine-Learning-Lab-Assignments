import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import RandomizedSearchCV


class MyPerceptron(BaseEstimator, ClassifierMixin):

    def __init__(self, learning_rate=0.05, max_epochs=1000):
        self.learning_rate = learning_rate
        self.max_epochs = max_epochs

    def sigmoid_function(self, sum):
        return 1 / (1 + np.exp(-sum))

    def fit(self, X, y):

        w0 = 10
        w1 = 0.2
        w2 = -0.75

        n = self.learning_rate

        for epoch in range(1, self.max_epochs + 1):

            errors = []

            for x, t in zip(X, y):

                k = x[0]
                l = x[1]

                total = k * w1 + l * w2 + w0

                pred = self.sigmoid_function(total)

                err = t - pred
                errors.append(err)

                delta_w0 = -n * (pred - t) * 1
                delta_w1 = -n * (pred - t) * k
                delta_w2 = -n * (pred - t) * l

                w0 = w0 + delta_w0
                w1 = w1 + delta_w1
                w2 = w2 + delta_w2
            sse = sum(error ** 2 for error in errors)

            if sse <= 0.002:
                break

        self.w0_ = w0
        self.w1_ = w1
        self.w2_ = w2

        return self

    def predict(self, X):
        predictions = []
        for x in X:
            total = (
                x[0] * self.w1_
                + x[1] * self.w2_
                + self.w0_
            )
            pred = self.sigmoid_function(total)

            if pred >= 0.5:
                predictions.append(1)
            else:
                predictions.append(0)

        return np.array(predictions)

model = MyPerceptron()

X = np.array([[0, 0],[0, 1],[1, 0],[1, 1]])
y = np.array([0, 0, 0, 1])

param_grid = {
    'learning_rate': [0.001, 0.01, 0.05, 0.1, 0.5],
    'max_epochs': [100, 500, 1000, 2000]
}

random_search = RandomizedSearchCV(model,param_distributions=param_grid,n_iter=5,cv=3,random_state=42)
random_search.fit(X, y)
print("best parameters:")
print(random_search.best_params_)
print("best CV score:")
print(random_search.best_score_)