import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

from lime.lime_tabular import LimeTabularExplainer



# 1. LOAD DATASET


df = pd.read_csv("features.csv")

print("Dataset shape:", df.shape)
print(df.head())



# 2. SEPARATE FEATURES AND TARGET


X = df.drop(columns=['person_id', 'image_name'])
y = df['person_id']



# 3. ENCODE PERSON IDs


label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

print("\nNumber of classes:", len(label_encoder.classes_))



# 4. TRAIN-TEST SPLIT


X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.3,
    random_state=42,
    stratify=y
)



# 5. CREATE PIPELINE


pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('mlp', MLPClassifier(
        hidden_layer_sizes=(50,),
        max_iter=1000,
        random_state=42
    ))
])



# 6. TRAIN PIPELINE


print("\nTraining pipeline...")

pipeline.fit(X_train, y_train)

print("Training completed.")



# 7. TEST PIPELINE


y_pred = pipeline.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print("\nPipeline Accuracy:", accuracy)



# 8. CREATE LIME EXPLAINER


explainer = LimeTabularExplainer(
    X_train.values,
    feature_names=X_train.columns.tolist(),
    class_names=label_encoder.classes_.astype(str).tolist(),
    mode='classification',
    random_state=42
)



# 9. SELECT A TEST SAMPLE


sample_index = 0

sample = X_test.iloc[sample_index].values

actual_class = y_test[sample_index]
predicted_class = pipeline.predict(
    X_test.iloc[[sample_index]]
)[0]


print("\nActual class:",
      label_encoder.inverse_transform([actual_class])[0])

print("Predicted class:",
      label_encoder.inverse_transform([predicted_class])[0])



# 10. CREATE LIME EXPLANATION


explanation = explainer.explain_instance(
    sample,
    pipeline.predict_proba,
    num_features=10,
    top_labels=5
)



# 11. DISPLAY EXPLANATION


print("\nLIME Explanation:")

print(
    explanation.as_list(
        label=predicted_class
    )
)



# 12. DISPLAY IN BROWSER

explanation.save_to_file("lime_explanation.html")

print("\nLIME explanation saved as lime_explanation.html")