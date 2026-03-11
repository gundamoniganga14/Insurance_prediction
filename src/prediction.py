# 1. load scaler.pkl and model.pkl files
#2. create a function which will take input as list and return the predictio

import pickle
import numpy as np

class Insurance_Predictor:
    def __init__(self):
        with open('artifacts/scaler.pkl','rb') as f:
            self.scaler=pickle.load(f)
        with open('artifacts/model.pkl','rb') as f:
            self.model=pickle.load(f)

    def predict(self,Age,Annual_Income_LPA,Policy_Term_Years,Sum_Assured_Lakhs):
        input_data=np.array([[Age,Annual_Income_LPA,Policy_Term_Years,Sum_Assured_Lakhs]])
        # input_data is already 2D (1 row, 4 features); scaler expects 2D, not 3D
        input_scaled=self.scaler.transform(input_data)
        result=self.model.predict(input_scaled)
        return result[0]