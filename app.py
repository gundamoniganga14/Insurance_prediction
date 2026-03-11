import streamlit as st
from  src.prediction import Insurance_Predictor

st.title("Insurance  Prediction")
st.write("This is a simple insurance prediction web app using streamlit and sklearn")

Age=st.number_input("Enter age:")
Annual_Income_LPA=st.number_input("Enter Annual Income in LPA")
Policy_Term_Years=st.number_input("Enter Policy Term in Years")
Sum_Assured_Lakhs=st.number_input("Enter Sum Assured in Lakhs")

if st.button("Predict"):
    model=Insurance_Predictor()
    result = model.predict(Age, Annual_Income_LPA, Policy_Term_Years, Sum_Assured_Lakhs)
    st.success(result)