import streamlit as st
import pandas as pd
import numpy as np
import pickle


st.title('Diabetes Prediction App')

st.write(""" 
This app predicts the **Diabetes** of a person!
""")
st.write('---')

# Reads in saved classification model
load_clf = pickle.load(open('model.pkl', 'rb'))


def user_input_features():

    Age = st.sidebar.slider('Age', 16, 90, 25)
    Gender=st.sidebar.selectbox('Gender',('Male','Female'))
    Polyuria = st.sidebar.selectbox('Polyuria (Excessive urination)', ('No', 'Yes'))
    Polydipsia = st.sidebar.selectbox('Polydipsia (Excessive thirst)', ('No', 'Yes'))
    sudden_weight_loss = st.sidebar.selectbox('sudden weight loss', ('No', 'Yes'))
    weakness = st.sidebar.selectbox('weakness', ('No', 'Yes'))
    Polyphagia = st.sidebar.selectbox('Polyphagia (Insatiable hunger)', ('No', 'Yes'))
    Genital_thrush = st.sidebar.selectbox('Genital thrush', ('No', 'Yes'))
    visual_blurring = st.sidebar.selectbox('visual blurring', ('No', 'Yes'))
    Itching = st.sidebar.selectbox('Itching', ('No', 'Yes'))
    Irritability = st.sidebar.selectbox('Irritability', ('No', 'Yes'))
    delayed_healing = st.sidebar.selectbox('delayed healing', ('No', 'Yes'))
    partial_paresis = st.sidebar.selectbox('partial paresis', ('No', 'Yes'))
    muscle_stiffness = st.sidebar.selectbox('muscle stiffness', ('No', 'Yes'))
    Alopecia = st.sidebar.selectbox('Alopecia (Hair loss)', ('No', 'Yes'))
    Obesity = st.sidebar.selectbox('Obesity', ('No', 'Yes'))
    
    #Map it all to 0 and 1
    Gender=apply_map(Gender,{'Male':1,'Female':0})
    Polyuria=apply_map(Polyuria,{'Yes':1,'No':0})
    Polydipsia=apply_map(Polydipsia,{'Yes':1,'No':0})
    sudden_weight_loss=apply_map(sudden_weight_loss,{'Yes':1,'No':0})
    weakness=apply_map(weakness,{'Yes':1,'No':0})
    Polyphagia=apply_map(Polyphagia,{'Yes':1,'No':0})
    Genital_thrush=apply_map(Genital_thrush,{'Yes':1,'No':0})
    visual_blurring=apply_map(visual_blurring,{'Yes':1,'No':0}) 
    Itching=apply_map(Itching,{'Yes':1,'No':0})
    Irritability=apply_map(Irritability,{'Yes':1,'No':0}) 
    delayed_healing=apply_map(delayed_healing,{'Yes':1,'No':0})
    partial_paresis=apply_map(partial_paresis,{'Yes':1,'No':0})
    muscle_stiffness=apply_map(muscle_stiffness,{'Yes':1,'No':0})
    Alopecia=apply_map(Alopecia,{'Yes':1,'No':0})
    Obesity=apply_map(Obesity,{'Yes':1,'No':0})
    
    
    data = { 'Gender': Gender, 'Polyuria': Polyuria, 'Polydipsia': Polydipsia, 'sudden weight loss': sudden_weight_loss, 'weakness': weakness, 'Polyphagia': Polyphagia, 'Genital thrush': Genital_thrush, 'visual blurring': visual_blurring, 'Itching': Itching, 'Irritability': Irritability, 'delayed healing': delayed_healing, 'partial paresis': partial_paresis, 'muscle stiffness': muscle_stiffness, 'Alopecia': Alopecia, 'Obesity': Obesity,'Age': Age,}
    
    return data

def apply_map(x, map_dict):
    if x in map_dict.keys():
        return map_dict[x]
    else:
        return x
    

data= user_input_features()
df=pd.DataFrame(data, index=[0])
    
st.subheader('User Input parameters')
st.write(df)
 
prediction = load_clf.predict(df)

prediction=apply_map(prediction[0],{1:'Positive',0:'Negative'})

prediction_proba = load_clf.predict_proba(df)

st.subheader('Prediction:')
st.write(prediction)

st.subheader('Prediction Probability:')
st.write('The probability of the person having diabetes is %:')
st.write('Positive: ', round(prediction_proba[0][1]*100, 2))
st.write('Negative:', round(prediction_proba[0][0]*100, 2))


st.write('---')

#? How to run the app
#? streamlit run app.py
#? http://localhost:8501

