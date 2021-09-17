import streamlit as st
import pandas as pd
import numpy as np
import joblib
import sklearn

gender_dict = {0: 'Female', 1: 'Male'}
commute_dict = {
    '0-1 Miles': 0,
    '1-2 Miles': 1,
    '2-5 Miles': 2,
    '5-10 Miles': 3,
    '10+ Miles': 4
}
education_dict = {
    'Partial High School': 0,
    'High School': 1,
    'Partial College': 2,
    'Bachelors': 3,
    'Graduate Degree': 4
}
marital_dict = {0: 'Married', 1: 'Single'}
occupation_dict = {0: 'Clerical',
 1: 'Management',
 2: 'Manual',
 3: 'Professional',
 4: 'Skilled Manual'}
region_dict = {0: 'Europe', 1: 'North America', 2: 'Pacific'}

st.title("Motorcycle Purchasing Classification")

st.write("""
### By Jonathan Okeke
""")

st.write("""
### Would you buy a motorcycle ?
""")

# Sidebar options
st.sidebar.write("""
## Fill in your details below
""")
age = st.sidebar.slider("Age", 20, 70)
gender = st.sidebar.selectbox("Gender", ('Male', 'Female'))
marital = st.sidebar.selectbox('Marital Status', ('Married', 'Single'))
children = st.sidebar.slider("Number of children you have", 0,5)
region = st.sidebar.selectbox("Which region do you stay ?", ('Pacific', 'Europe', 'North America'))
cars = st.sidebar.slider("Number of cars you own", 0, 4)
home = st.sidebar.selectbox("Do you own your home ?", ('Yes' , 'No'))
income = st.sidebar.slider("Income", 10000, 200000)
education = st.sidebar.selectbox("Education Level" ,('Partial High School', 'High School', 'Partial College', 'Bachelors', 'Graduate Degree'))
occupation = st.sidebar.selectbox("Occupation" , ('Manual', 'Clerical', 'Skilled Manual', 'Professional', 'Management'))
commute = st.sidebar.selectbox("Commute Distance", ('0-1 Miles', '1-2 Miles', '2-5 Miles', '5-10 Miles', '10+ Miles'))


def predict(age, gender, marital, children, region, cars, home, education, occupation, commute, income):
  # Map values to encodings
  X = []
  if marital == 'Married':
    X.append(0)
  else:
    X.append(1)
  if gender == 'Female':
    X.append(0)
  else:
    X.append(1)
  X.append(income)
  X.append(children)
  for key,value in education_dict.items():
    if education in key:
      X.append(value)
      break
  for key,value in occupation_dict.items():
    if occupation in value:
      X.append(key)
      break
  if home == 'Yes':
    X.append(1)
  else:
    X.append(0)
  X.append(cars)
  for key, value in commute_dict.items():
    if commute in key:
      X.append(value)
  for key,value in region_dict.items():
    if region in value:
      X.append(key)
  X.append(age)

  arr = np.array(X).reshape(-1, 11)
  # Load the model
  model = joblib.load('rfc_model_v1.mdl')
  prediction = model.predict(arr)
  prob = model.predict_proba(arr)

  return prediction , prob


# Load model and predict

prediction, prob  = predict(age, gender, marital, children, region, cars, home, education, occupation, commute, income)

if prediction[0] == 1:
  prediction = 'Yes'
else:
  prediction = 'No'

st.write(f"Prediction: {prediction}")
st.write(f"Probability: {str(round(100 * prob.max(), 2))} %")