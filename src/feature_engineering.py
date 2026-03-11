# 1.load training  and testing data
# 2.feature scaling the training and testing data
#3.save the scaled data into processed folder csv files
import pickle
from data_preprocessing import load_and_split_data
import pandas as pd
from sklearn.preprocessing import StandardScaler

# 2.feature scaling the training and testing data
x_train,x_test,y_train,y_test=load_and_split_data()

scaler=StandardScaler()

x_train_scaled=scaler.fit_transform(x_train)
x_test_scaled=scaler.transform(x_test)

#3.save the scaled data into processed folder csv files
pd.DataFrame(x_train_scaled).to_csv('../data/processed/x_train.csv',index=False)
pd.DataFrame(x_test_scaled).to_csv('../data/processed/x_test.csv',index=False)
pd.DataFrame(y_train).to_csv('../data/processed/y_train.csv',index=False)   
pd.DataFrame(y_test).to_csv('../data/processed/y_test.csv',index=False)

with open('../artifacts/scaler.pkl','wb') as f:
    pickle.dump(scaler,f)