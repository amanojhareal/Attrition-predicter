import pandas as pd
import numpy as np

df =  pd.read_csv("../data/raw/WA_Fn-UseC_-HR-Employee-Attrition.csv")

df = pd.get_dummies(df, drop_first=True, dtype = int)


