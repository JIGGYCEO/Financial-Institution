import pandas as pd
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
import pickle

data = pd.read_csv('Financial_inclusion_dataset.csv')
data.head()

df = data.copy()
df.drop('uniqueid', inplace = True, axis = 1)

df = df.drop_duplicates()

cat = df.select_dtypes(include = ['object', 'category'])
num = df.select_dtypes(include= 'number')

from sklearn.preprocessing import StandardScaler, LabelEncoder
scaler = StandardScaler()
encoder = LabelEncoder()

for i in num.columns: 
    if i in df.columns: 
        df[i] = scaler.fit_transform(df[[i]]) 
for i in cat.columns:
    if i in df.columns: 
        df[i] = encoder.fit_transform(df[i])

# - Using XGBOOST to find feature importance
x = df.drop('bank_account', axis = 1)
y = df.bank_account 

import xgboost as xgb
model = xgb.XGBClassifier()
model.fit(x, y)

# Print feature importance scores
xgb.plot_importance(model)

sel_cols = ['country', 'year', 'location_type', 'cellphone_access',
       'household_size', 'age_of_respondent', 'gender_of_respondent',
       'relationship_with_head', 'marital_status', 'education_level',
       'job_type']
x = df[sel_cols]
x = pd.concat([x, df['bank_account']], axis = 1)



# UnderSampling The Majority Class
class1 = df.loc[df['bank_account'] == 1]    
class0 = df.loc[df['bank_account'] == 0]   
class1_3000 = class0.sample(5000)   
new_dataframe = pd.concat([class1_3000, class1], axis = 0) 


# #---------------MODELLING--------------------
x = new_dataframe.drop('bank_account', axis = 1)
y = new_dataframe['bank_account']

from sklearn.model_selection import train_test_split
xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size = 0.20, random_state = 40, stratify = y)

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

model = RandomForestClassifier() 
model.fit(xtrain, ytrain) 
cross_validation = model.predict(xtrain)
pred = model.predict(xtest) 

# save model
model = pickle.dump(model, open('Financial_inclusion.pkl', 'wb'))
print('\nModel is saved\n')


#-----------------------STREAMLIT DEVELOPMENT----------------------------------

model = pickle.load(open('Financial_inclusion.pkl','rb'))

st.markdown("<h1 style = 'color: #00092C; text-align: center;font-family: Arial, Helvetica, sans-serif; '>FINANCIAL INCLUSION</h1>", unsafe_allow_html= True)
st.markdown("<h3 style = 'margin: -25px; color: #45474B; text-align: center;font-family: Arial, Helvetica, sans-serif; '> Interactive Insights into Banking and Financial Access</h3>", unsafe_allow_html= True)
st.image('image 1.png', width = 600)
st.markdown("<h2 style = 'color: #0F0F0F; text-align: center;font-family: Arial, Helvetica, sans-serif; '>BACKGROUND OF STUDY </h2>", unsafe_allow_html= True)

st.markdown('<br2>', unsafe_allow_html= True)

st.markdown("<p>The endeavor to guarantee that all individuals and businesses, irrespective of their financial status, have access to basic financial services such as credit, banking, and insurance is known as financial inclusion. Giving people the tools they need to manage their money wisely—saving, borrowing, and risk-averse—is the main objective, especially for those living in underprivileged areas. Financial inclusion seeks to advance economic development, lessen poverty, and create a more robust and comprehensive financial system by improving accessibility, affordability, and technological innovation. This project aims to identify the people who are most likely to have or use a bank account.</p>",unsafe_allow_html= True)


st.sidebar.image('image 2.png')

dx = data[['country', 'year', 'location_type', 'cellphone_access',
       'household_size', 'age_of_respondent', 'gender_of_respondent',
       'relationship_with_head', 'marital_status', 'education_level',
       'job_type']]

st.write(data.head())

age_of_respondent = st.sidebar.number_input("age_of_respondent", data['age_of_respondent'].min(), data['age_of_respondent'].max())
household_size = st.sidebar.number_input("household_size", data['household_size'].min(), data['household_size'].max())
job_type = st.sidebar.selectbox("Job Type", data['job_type'].unique())
education_level = st.sidebar.selectbox("education_level", data['education_level'].unique())
marital_status = st.sidebar.selectbox("marital_status", data['marital_status'].unique())
country = st.sidebar.selectbox('country', data['country'].unique())
year = st.sidebar.number_input("year", data['year'].min(), data['year'].max())
location_type = st.sidebar.selectbox("location_type", data['location_type'].unique())
cellphone_access = st.sidebar.selectbox("cellphone_access", data['cellphone_access'].unique())       
gender_of_respondent = st.sidebar.selectbox("gender_of_respondent", data['gender_of_respondent'].unique())
relationship_with_head = st.sidebar.selectbox("relationship_with_head", data['relationship_with_head'].unique())

# ['country', 'year', 'location_type', 'cellphone_access',
#        'household_size', 'age_of_respondent', 'gender_of_respondent',
#        'relationship_with_head', 'marital_status', 'education_level',
#        'job_type']

# Bring all the inputs into a dataframe
input_variable = pd.DataFrame([{
    'country': country,
    'year' : year,
    'location_type' : location_type,
    'cellphone_access' : cellphone_access,
    'household_size': household_size,
    'age_of_respondent': age_of_respondent,
    'gender_of_respondent' : gender_of_respondent,
    'relationship_with_head' : relationship_with_head,
    'marital_status': marital_status,
    'education_level': education_level,
    'job_type': job_type,   
}])

# Reshape the Series to a DataFrame
# input_variable = input_data.to_frame().T

st.write(input_variable)

cat = input_variable.select_dtypes(include = ['object', 'category'])
num = input_variable.select_dtypes(include = 'number')
# Standard Scale the Input Variable.

from sklearn.preprocessing import StandardScaler, LabelEncoder

for i in num.columns:
    if i in input_variable.columns:
      input_variable[i] = StandardScaler().fit_transform(input_variable[[i]])
for i in cat.columns:
    if i in input_variable.columns: 
        input_variable[i] = LabelEncoder().fit_transform(input_variable[i])

st.markdown('<hr>', unsafe_allow_html=True)
st.markdown("<h2 style = 'color: #0A2647; text-align: center; font-family: helvetica '>Model Report</h2>", unsafe_allow_html = True)

if st.button('Press To Predict'):
    predicted = model.predict(input_variable)
    st.toast('bank_account Predicted')
    st.image('image.jpg', width = 100)
    st.success(f'Model Predicted {predicted}')
    if predicted == 0:
        st.success('The person does not have an account')
    else:
        st.success('the person has an account')

st.markdown('<hr>', unsafe_allow_html=True)
st.markdown("<h8>FINANCIAL INCLUSION built by Adekunle Mojeed</h8>", unsafe_allow_html=True)
