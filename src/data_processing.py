import pandas as pd
import numpy as np

df =  pd.read_csv("../data/raw/WA_Fn-UseC_-HR-Employee-Attrition.csv")

drop_cols = ['Over18','EmployeeCount','EmployeeNumber','StandardHours']
df = df.drop(columns=drop_cols)
categorical_cols = df.select_dtypes(include='object').columns.to_list()

one_hot_cols =[]
map_cols =[]

for col in categorical_cols:
    unique_count = df[col].nunique()
    
    if unique_count == 2:
        map_cols.append(col)
        
    
    else :
        one_hot_cols.append(col)
        


yes_no_cols = ['Attrition','OverTime']
for col in yes_no_cols:
    df[col]=df[col].map({
        'Yes':1,
        'No':0
    })
    
df['Gender'] = df['Gender'].map({
    'Male':1,
    'Female':0
})
df = pd.get_dummies(df,columns=one_hot_cols,drop_first=True,dtype=int)

df.to_csv("processed_data1.csv", index=False)
print("data processed and saved✅")

