from sklearn.neural_network import MLPClassifier

X = [[0, 0],
     [0, 1],
     [1, 0],
     [1, 1]]


y = [0, 0, 0, 1]

clf = MLPClassifier(
    solver='lbfgs',
    alpha=1e-5,
    hidden_layer_sizes=(2,),
    random_state=42,
    max_iter=1000
)

clf.fit(X, y)

print("AND:", clf.predict([[1, 1], [0, 1]]))

y = [0, 1, 1, 0]

clf2 = MLPClassifier(
    solver='lbfgs',
    alpha=1e-5,
    hidden_layer_sizes=(2,),
    random_state=42,
    max_iter=1000
)

clf2.fit(X, y)

print("XOR:", clf2.predict([[1, 1], [0, 1]]))