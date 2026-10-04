import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

from sklearn.linear_model import Perceptron
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from xgboost import XGBClassifier
from catboost import CatBoostClassifier




df = pd.read_csv("features.csv")


X = df.drop(columns=['person_id', 'image_name'])
y = df['person_id']

label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(X,y_encoded,test_size=0.3,random_state=42,stratify=y)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model_perceptron=Perceptron(max_iter=1000,random_state=42)
model_mlp=MLPClassifier(hidden_layer_sizes=(50,),max_iter=1000,random_state=42)
model_svm=SVC(random_state=42)
model_Decision=DecisionTreeClassifier(random_state=42)
model_random= RandomForestClassifier( n_estimators=100,random_state=42)
model_Cat= CatBoostClassifier(iterations=100,depth=4,learning_rate=0.1,verbose=0,random_state=42)
model_ada=AdaBoostClassifier(random_state=42)
model_xgboost=XGBClassifier(eval_metric='mlogloss', random_state=42)
model_naive=GaussianNB()

results = []

model_perceptron.fit(X_train_scaled,y_train)
model_mlp.fit(X_train_scaled,y_train)
model_svm.fit(X_train_scaled,y_train)
model_naive.fit(X_train_scaled,y_train)
model_Decision.fit(X_train,y_train)
model_random.fit(X_train,y_train)
model_Cat.fit(X_train,y_train)
model_ada.fit(X_train,y_train)
model_xgboost.fit(X_train,y_train)

y_pred_perceptron=model_perceptron.predict(X_test_scaled)
y_pred_mlp=model_mlp.predict(X_test_scaled)
y_pred_svm=model_svm.predict(X_test_scaled)
y_pred_naive=model_naive.predict(X_test_scaled)
y_pred_Decision=model_Decision.predict(X_test)
y_pred_random=model_random.predict(X_test)
y_pred_Cat=model_Cat.predict(X_test)
y_pred_ada=model_ada.predict(X_test)
y_pred_xgboost=model_xgboost.predict(X_test)



predictions = {
    "Perceptron": y_pred_perceptron,
    "MLP": y_pred_mlp,
    "SVM": y_pred_svm,
    "Decision Tree": y_pred_Decision,
    "Random Forest": y_pred_random,
    "CatBoost": y_pred_Cat,
    "AdaBoost": y_pred_ada,
    "XGBoost": y_pred_xgboost,
    "Naive Bayes": y_pred_naive
}

results = []

for name, y_pred in predictions.items():
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test,y_pred,average='weighted',zero_division=0)
    recall = recall_score(y_test,y_pred,average='weighted',zero_division=0)
    f1 = f1_score(y_test, y_pred,average='weighted',zero_division=0)
    results.append({
        "Classifier": name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1-Score": f1
    })


results_df = pd.DataFrame(results)

results_df[
    ["Accuracy", "Precision", "Recall", "F1-Score"]
] = results_df[
    ["Accuracy", "Precision", "Recall", "F1-Score"]
].round(4)


print("\nFinal Results:")
print(results_df)