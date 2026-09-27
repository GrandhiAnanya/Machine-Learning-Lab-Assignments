import pandas as pd
import numpy as np 
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

def load_data():
    df = pd.read_excel(
        "Lab Session Data.xlsx",
        sheet_name="Purchase data"
    )
    return df

def customer_class(payment):
    if payment>200:
        return "RICH"
    return "POOR"

df=load_data()
df["Class"] = df["Payment (Rs)"].apply(customer_class)

X = df[[
    "Candies (#)",
    "Mangoes (Kg)",
    "Milk Packets (#)"
]]

y = df["Class"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

model = DecisionTreeClassifier(random_state=42)
model.fit(X_train,y_train)

predictions=model.predict(X_test)

accuracy = accuracy_score(y_test,predictions)
print("Accuracy:",accuracy)

all_predictions = model.predict(X)

result = pd.DataFrame({
    "Customer": df["Customer"],
    "Candies (#)": df["Candies (#)"],
    "Mangoes (Kg)": df["Mangoes (Kg)"],
    "Milk Packets (#)": df["Milk Packets (#)"],
    "Actual Class": df["Class"],
    "Predicted Class": all_predictions
})

print(result)