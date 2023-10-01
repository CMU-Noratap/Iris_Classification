import pickle
import warnings

import streamlit as st

warnings.filterwarnings("ignore")
from PIL import Image

labels = ["Species_Iris-setosa","Species_Iris-versicolor","Species_Iris-virginica"]

pickle_in = open("model/rf_model.pkl","rb")
classifier = pickle.load(pickle_in)

file_std = open("model/std_scaler.pkl","rb")
std_scaler = pickle.load(file_std)

def predict(sepal_l,sepal_w,petal_l,petal_w):
    x_scale =  std_scaler.transform([[sepal_l,sepal_w,petal_l,petal_w]])
    y_pred = classifier.predict(x_scale)
    print(y_pred)
    return y_pred

def Input_Output():
    st.title("Iris Variety Prediction")
    
    st.markdown("You are using Streamlit...",unsafe_allow_html=True)
    sepal_l = st.text_input("Enter Sepal Length",".")
    sepal_w = st.text_input("Enter Sepal Width",".")
    petal_l = st.text_input("Enter Petal Length",".")
    petal_w = st.text_input("Enter Petal Width",".")

    result = ""
    if st.button("Predict"):
        result = predict(sepal_l,sepal_w,petal_l,petal_w)
        st.balloons()
    st.success(f"The output is : {result}")

if __name__ == '__main__':
    Input_Output()