import pandas as pd
import numpy as np
import shap

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score


df = pd.read_csv("features.csv")

print("Dataset shape:", df.shape)
print(df.head())


X = df.drop(columns=['person_id', 'image_name'])
y = df['person_id']

label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

print("\nNumber of classes:", len(label_encoder.classes_))
print("Classes:", label_encoder.classes_)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.3,
    random_state=42,
    stratify=y
)

pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('mlp', MLPClassifier(
        hidden_layer_sizes=(50,),
        max_iter=1000,
        random_state=42
    ))
])

print("\nTraining MLP pipeline...")

pipeline.fit(X_train, y_train)

print("Training completed.")

y_pred = pipeline.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print("\nAccuracy:", accuracy)


print("\nCreating SHAP explainer...")

background = X_train.sample(
    50,
    random_state=42
)

explainer = shap.Explainer(
    pipeline.predict_proba,
    background
)


print("Calculating SHAP values...")

X_explain = X_test.iloc[:20]

shap_values = explainer(X_explain)

print("SHAP values calculated successfully.")

print("SHAP values shape:", shap_values.values.shape)

class_index = 0

class_name = label_encoder.inverse_transform(
    [class_index]
)[0]

print("\nExplaining class:", class_name)
class_shap_values = shap_values[:, :, class_index]

print(
    "Selected class SHAP shape:",
    class_shap_values.values.shape
)

print("\nDisplaying SHAP bar plot...")

shap.plots.bar(
    class_shap_values
)

print("\nDisplaying SHAP beeswarm plot...")

shap.plots.beeswarm(
    class_shap_values
)