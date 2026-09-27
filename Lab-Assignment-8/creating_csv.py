import pandas as pd

df = pd.DataFrame({
    "Customer": ["C_1", "C_2", "C_3", "C_4", "C_5",
                 "C_6", "C_7", "C_8", "C_9", "C_10"],

    "Candies": [20, 16, 27, 19, 24, 22, 15, 18, 21, 16],

    "Mangoes": [6, 3, 6, 1, 4, 1, 4, 4, 1, 2],

    "Milk_Packets": [2, 6, 2, 2, 2, 5, 2, 2, 4, 4],

    "Payment": [386, 289, 393, 110, 280, 167, 271, 274, 148, 198],

    "High_Value": [1, 1, 1, 0, 1, 0, 1, 1, 0, 0]
})

df.to_csv("customer_data.csv", index=False)

print(df)