import pandas as pd
import matplotlib.pyplot as plt

def load_data():
    df=pd.read_excel(
        "Lab Session Data.xlsx",
        sheet_name="thyroid0387_UCI")
    return df

def calculate_stats(df):
    mean_age=df["age"].mean()
    var_age=df["age"].var()
    std_age=df["age"].std()

    return mean_age,var_age,std_age



df=load_data()
#print(df.info())
'''this command shows the number of columns, the number of non null values
 and the datatype according to which i was able to identify the 
 type of enocding to use '''

  
# Encoding Scheme for Categorical Attributes
# Binary attributes (on thyroxine, sick, pregnant, etc.) -> Label Encoding
# sex -> One-Hot Encoding
# referral source -> One-Hot Encoding
# Condition (Target Variable) -> Label Encoding

print(df.describe())
"""observation : - the maximum age is 65526 which is not possible indicating there are erroneous values or outliers 
                   other numerical attributes such as TSH,T3,TT4,T4U,FTI AND TBG are not included as they are stored as object datatypes due to the presence of non-numeric values
"""
print("\n")
#print(df.isnull().sum())
print((df == "?").sum())
""" observation : - No missing values are detected using isnull() because the dataset stores
                    missing values are taken as '?' instead of NaN."""



plt.boxplot(df["age"])
plt.title("Box Plot of Age")
plt.ylabel("Age")
plt.show()
"""the box plot shows the presence of outliers in the age attribute"""

mean,var,std=calculate_stats(df)

print("\nAge Statistics")
print("Mean :", mean)
print("Variance :", var)
print("Standard Deviation :", std)