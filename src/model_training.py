
# 1. load processed data from processed folder
# 2 . create a model and train the data
#3.save the model into model folder
import pickle
import pandas as pd
from sklearn.linear_model import LinearRegression
x_train=pd.read_csv('../data/processed/x_train.csv')
x_test=pd.read_csv('../data/processed/x_test.csv')
y_train=pd.read_csv('../data/processed/y_train.csv')
y_test=pd.read_csv('../data/processed/y_test.csv')

print(x_train)

model=LinearRegression()
model.fit(x_train,y_train)

with open('../artifacts/model.pkl','wb') as f:
    pickle.dump(model,f)