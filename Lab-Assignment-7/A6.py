import pandas as pd
import matplotlib.pyplot as plt

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.tree import plot_tree



def load_data():
    df = pd.read_csv("features.csv")
    return df



df = load_data()
X = df.drop(columns=['person_id', 'image_name'])
y = df['person_id']
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3,random_state=1)

clf = DecisionTreeClassifier(random_state=1)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
plt.figure(figsize=(25, 15))
plot_tree(clf,feature_names=X.columns,class_names=[str(c) for c in clf.classes_],filled=True,rounded=True)
plt.title("Decision Tree")
plt.savefig("decision_tree.png",dpi=300,bbox_inches="tight")
plt.show()