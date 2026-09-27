from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
import pandas as pd

def load_data():
    df=pd.read_csv("features.csv")
    return df

df=load_data()
X = df.drop(columns=['person_id', 'image_name'])
y = df['person_id']

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=42,stratify=y)

clf = MLPClassifier(
    solver='lbfgs',
    alpha=1e-5,
    hidden_layer_sizes=(50,),
    random_state=1,
    max_iter=5000
)

clf.fit(X_train, y_train)

predictions = clf.predict(X_test)

for prediction, actual in zip(predictions, y_test):
    print(f"Prediction: {prediction} actual value: {actual}")


