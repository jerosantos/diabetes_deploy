import numpy as np
import pandas as pd

# Import streamlit
import streamlit as st
# from sklearn.externals import joblib
import joblib

# Web app title
st.title('Diabetes')

# Describe
st.write('Legit na web app ito')

# Read data
data = pd.read_csv('diabetes.csv')

# Show data
data

# Let's draw a historgram
st.subheader('Age Distribution')

# Use numpy to generate bins for age
hist_values = np.histogram(data.Age, bins=100, range=(0,100))[0]

# Show bar chart
st.bar_chart(hist_values)

# Add sliders and assign them to variables
st.sidebar.subheader('Diabetes Factors')

# Age slider
age = st.sidebar.slider('Age', 0, 100, 30) # (Title, min value, max value, default value)

# BMI slider
bmi = st.sidebar.slider('Body mass index', 0, 70, 20)

# Plasma conc
plasma = st.sidebar.slider('Plasma glucose concentration', 0, 100, 50)

# Load saved ML model
st.subheader('Predicting Diabetes')

# Load model suing joblib
saved_model = joblib.load('finalized_model2.sav')

#Predict
predict_diabetes = saved_model.predict([[age, bmi, plasma]])[0]

# Print 
if predict_diabetes == 0:
    st.write('No Diabetes')
elif predict_diabetes == 1:
    st.write('Die abetes')