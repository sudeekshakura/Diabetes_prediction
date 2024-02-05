#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 31 15:40:48 2024

@author: sudeekshakura
"""

import numpy as np
# Loading and dumping model
import pickle
#streamlit is used for deployment
import streamlit as st

loaded_model= pickle.load(open('/Users/sudeekshakura/Desktop/ML Practice/project_3_streamlit/trained_model.sav','rb'))


# Creating a function for prediction

def diabeted_prediction (input_data):

    #Change input data to numpy array
    input_data_array=np.asarray(input_data)

    #reshape the numpy array
    input_array_reshaped= input_data_array.reshape(1,-1)
    prediction=loaded_model.predict(input_array_reshaped)
    if prediction[0]==0:
        return('Patient is non-diabetic.')
    else:
        return('Patient is diabetic.')
    

#streamlit function
def main():
    
    #Giving a title for our webpage
    st.title('Diabetes Prediction Web App')
    
    #Getting the input data from the user 
    Pregnancies= st.text_input('Number of Pregnancies')
    Glucose= st.text_input('Glucose Level')
    BloodPressure= st.text_input('BloodPressure')
    SkinThickness= st.text_input('SkinThickness')
    Insulin= st.text_input('Insulin Level')
    BMI= st.text_input('BMI value')
    DiabetesPedigreeFunction= st.text_input('Diabetes Pedigree Function')
    Age= st.text_input('Age of the person')
    
    #Code for prediction
    diagnosis=''
    
    #Create a button
    if st.button("Diabetes Test Result"):
        diagnosis=	diabeted_prediction([Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age])

    st.success(diagnosis)


if __name__=='__main__':
    main()		
    
    