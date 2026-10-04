from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import RandomizedSearchCV

X = [[0,0],[0,1],[1,0],[1,1]]
y = [0,0,0,1]

mlp = MLPClassifier(random_state=1)

param_grid = {
    'hidden_layer_sizes': [(2,), (5,), (10,), (5,5), (10,10)],
    'activation': ['relu', 'tanh', 'logistic'],
    'solver': ['lbfgs', 'adam'],
    'alpha': [0.0001, 0.001, 0.01, 0.1],
    'learning_rate_init': [0.001, 0.01, 0.1],
    'max_iter': [500, 1000, 2000]
}

random_search = RandomizedSearchCV(
    mlp,
    param_distributions=param_grid,
    n_iter=10,
    cv=3,
    random_state=42
)

random_search.fit(X, y)

print("Best parameters:")
print(random_search.best_params_)

print("Best score:")
print(random_search.best_score_)

best_model = random_search.best_estimator_

print("Predictions:")
print(best_model.predict(X))