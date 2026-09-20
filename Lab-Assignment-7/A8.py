import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split

def load_data():
    df = pd.read_csv("features.csv")
    return df

df = load_data()
X = df.drop(columns=['person_id', 'image_name'])
y = df['person_id']
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3,random_state=1, stratify=y) 
clf = DecisionTreeClassifier(random_state=1)
param_grid = {
    'criterion': ['gini', 'entropy'],
    'max_depth': [3, 5, 10, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 5]
}

grid_search = GridSearchCV(
    estimator=clf,
    param_grid=param_grid,
    cv=3,
    scoring='accuracy',
    n_jobs=-1
)

grid_search.fit(X_train, y_train)
mean_scores = grid_search.cv_results_['mean_test_score']
params = grid_search.cv_results_['params']

# Print each mean CV score with its corresponding parameters
for mean, param in zip(mean_scores, params):
    print(f"Mean: {mean:.4f} | Params: {param}")
print("Best Hyperparameters:")
print(grid_search.best_params_)
print("\nBest Cross-Validation Accuracy:")
print(grid_search.best_score_)
best_clf = grid_search.best_estimator_
print("\nTest Accuracy:")
print(best_clf.score(X_test, y_test))