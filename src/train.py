import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

df = pd.read_csv("../data/processed/processed_data1.csv")

X = df.drop(['Attrition'],axis=1)
y = df['Attrition']

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42,stratify=y)


scaler =StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = DecisionTreeClassifier(
    random_state=42,
    max_depth=5
)
model.fit(X_train,y_train)

joblib.dump(model,"../models/model.pkl")
joblib.dump(scaler, "../models/scaler.pkl")

print("model saved and scaler also")