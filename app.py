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
    Polyuria = st.sidebar.selectbox('Polyuria (Excessive urination)', ('Yes', 'No'))
    Polydipsia = st.sidebar.selectbox('Polydipsia (Excessive thirst)', ('Yes', 'No'))
    sudden_weight_loss = st.sidebar.selectbox('sudden weight loss', ('Yes', 'No'))
    weakness = st.sidebar.selectbox('weakness', ('Yes', 'No'))
    Polyphagia = st.sidebar.selectbox('Polyphagia (Insatiable hunger)', ('Yes', 'No'))
    Genital_thrush = st.sidebar.selectbox('Genital thrush', ('Yes', 'No'))
    visual_blurring = st.sidebar.selectbox('visual blurring', ('Yes', 'No'))
    Itching = st.sidebar.selectbox('Itching', ('Yes', 'No'))
    Irritability = st.sidebar.selectbox('Irritability', ('Yes', 'No'))
    delayed_healing = st.sidebar.selectbox('delayed healing', ('Yes', 'No'))
    partial_paresis = st.sidebar.selectbox('partial paresis', ('Yes', 'No'))
    muscle_stiffness = st.sidebar.selectbox('muscle stiffness', ('Yes', 'No'))
    Alopecia = st.sidebar.selectbox('Alopecia (Hair loss)', ('Yes', 'No'))
    Obesity = st.sidebar.selectbox('Obesity', ('Yes', 'No'))
    
    #Map it all to 0 and 1
    Gender=apply_map(Gender,{'Male':1,'Female:':0})
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
    
    features = pd.DataFrame(data, index=[0])
    
    
    return features

def apply_map(x, map_dict):
    if x in map_dict.keys():
        return map_dict[x]
    else:
        return x
    

df = user_input_features()

st.subheader('User Input parameters')
st.write(df)
 
prediction = load_clf.predict(df)

prediction=apply_map(prediction[0],{1:'Positive',0:'Negative'})

prediction_proba = load_clf.predict_proba(df)

st.subheader('Prediction:')
st.write(prediction)

st.subheader('Prediction Probability:')
st.write('The probability of the person having diabetes is %:')
st.write('Positive: ', prediction_proba[0][1])
st.write('Negative:', prediction_proba[0][0])


st.write('---')

#? How to run the app
#? streamlit run app.py
#? http://localhost:8501

