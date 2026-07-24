import numpy as np
import pandas as pd
import math

def load_data():
    df=pd.read_excel(
        "Lab Session Data.xlsx",
        sheet_name="marketing_campaign"
    )

    return df

def mean_var_sd(X):
    means=[]
    vars=[]
    sds=[]

    for j in range(len(X[0])):
        total_sum=0
        for i in range(len(X)):
            total_sum+=X[i][j]
        mean=total_sum/len(X)
        means.append(mean)

    
    
        var=0
        for i in range(len(X)):
            var+=(X[i][j]-mean)**2
        var=var/len(X)
        vars.append(var)

        sds.append(var**0.5)
   
        
    return means,vars,sds

def mean_var_sd_lib(X):
    mean=np.mean(X, axis=0)
    var=np.var(X,axis = 0)
    sd=np.std(X,axis=0)

    return mean,var,sd
   



df=load_data()
X = df.select_dtypes(include=np.number).to_numpy()
mean,var,sd=mean_var_sd(X)
mean1,var1,sd1=mean_var_sd_lib(X)

print(f"\n Mean value : {mean}")
print(f"\n Mean value using numpy : {mean1}")
print(f"\n Variance value : {var}")
print(f"\n Varaince value using numpy : {var1}")
print(f"\n Standar deviation value : {sd}")
print(f"\n Standar deviation value using numpy : {sd1}")



