import streamlit as st
st. set_page_config(layout="wide")

import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

import pandas as pd 
from matplotlib import pyplot as plt
from plotly import graph_objs as go
import numpy as np 
from tensorflow.keras.models import load_model
import joblib


data_p           = pd.read_csv("data//proximate.csv")
data_u           = pd.read_csv("data//ultimate.csv")

proximate        = load_model('utils/model_proximate_scaled.h5', compile = False)
ultimate         = load_model('utils/model_ultimate_scaled.h5', compile = False)

proximate_scaler = joblib.load("utils/scaler_proximate.pkl")
ultimate_scaler  = joblib.load("utils/ULTIMATE_SCALER.pkl")

proximate_rnd    = joblib.load("utils/RND _FRST_REG_SCALED_PROXIMATE.pkl")
proximate_lr     = joblib.load("utils/LINEAR_REG_SCALED_PROXIMATE.pkl")

ultimate_rnd     = joblib.load("utils/RND_FRST_REG_SCALED_ULTIMATE.pkl")
ultimate_lr      = joblib.load("utils/LINEAR_REG_SCALED_ULTIMATE.pkl")

st.title("HHV PREDICTOR")
st.image("data//appImage.jpg",width = 500)
nav = st.sidebar.radio("Navigation",["Home","Prediction"])
if nav == "Home":
    
    if st.checkbox("Show Table"):
        col1, col2 = st.columns(2)
        col1.table(data_p.head(10))
        col2.table(data_u.head(10))

    
    graph = st.selectbox("What kind of Graph ? ",["Non-Interactive","Interactive"])

    if graph == "Non-Interactive":
        fig = plt.figure(figsize = (6,3))
        plt.subplot(121)
        plt.scatter(data_p["FC"],data_p["HHV"])
        plt.xlabel("FC")
        plt.ylabel("HHV")

        plt.subplot(122)
        plt.scatter(data_u["C"],data_u["HHV"])
        plt.xlabel("C")
        plt.ylabel("HHV")
        plt.tight_layout()
        st.pyplot(fig)

    if graph == "Interactive":
        col1, col2 = st.columns(2)
        layout =go.Layout(
            xaxis = dict(range=[np.min(data_p['FC']),np.max(data_p['FC'])]),
            yaxis = dict(range =[np.min(data_p['HHV']),np.max(data_p['HHV'])])
        )
        fig = go.Figure(data=go.Scatter(x=data_p["FC"], y=data_p["HHV"], mode='markers'),layout = layout)
        col1.plotly_chart(fig)

        layout2 =go.Layout(
            xaxis = dict(range=[np.min(data_u['C']),np.max(data_u['C'])]),
            yaxis = dict(range =[np.min(data_u['HHV']),np.max(data_u['HHV'])])
        )
        fig = go.Figure(data=go.Scatter(x=data_u["C"], y=data_u["HHV"], mode='markers'),layout = layout2)
        col2.plotly_chart(fig)
    
if nav == "Prediction":
    st.header("CALCULATE HHV")
    ana = st.radio("select",["proximate","ultimate"])
    if ana == 'proximate':
        mod = st.radio("SELECT MODEL", ["ANN", "LR", "RND_FRST"])

        val1 = st.number_input('VM', 1.0)
        val2 = st.number_input('FC', 1.0)
        val3 = st.number_input('ASH', 0.1)

        val = np.array([[val1,val2,val3]])
        val = proximate_scaler.transform(val)
        

        if st.button("Predict"):
            if mod == "ANN":
                pred =proximate.predict(val.reshape(1,3,1))
                st.success(f"Your predicted HHV is {pred}")
            elif mod == "LR":
                pred =proximate_lr.predict(val)
                st.success(f"Your predicted HHV is {pred}")
            else:
                pred =proximate_rnd.predict(val)
                st.success(f"Your predicted HHV is {pred}")

    elif ana == 'ultimate':
        mod = st.radio("SELECT MODEL", ["ANN", "LR", "RND_FRST"])

        val1 = st.number_input('C', 1.0)
        val2 = st.number_input('H', 1.0)
        val3 = st.number_input('N', 0.1)
        val4 = st.number_input('O', 1.0)
        val5 = st.number_input('S', 0.0)


        val = np.array([[val1,val2,val3,val4,val5]])
        val = ultimate_scaler.transform(val)
        

        if st.button("Predict"):
            if mod == "ANN":
                pred =ultimate.predict(val.reshape(1,5,1))
                st.success(f"Your predicted HHV is {pred}")
            elif mod == "LR":
                pred =ultimate_lr.predict(val)
                st.success(f"Your predicted HHV is {pred}")
            else:
                pred =ultimate_rnd.predict(val)
                st.success(f"Your predicted HHV is {pred}")
            
