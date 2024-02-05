# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""


import numpy as np
import pickle

loaded_model= pickle.load(open('/Users/sudeekshakura/Desktop/ML Practice/project_3_streamlit/trained_model.sav','rb'))

#Making a predictive system 

input_data=(4,110,92,0,0,37.6,0.191,30)

#Change input data to numpy array
input_data_array=np.asarray(input_data)

#reshape the numpy array
input_array_reshaped= input_data_array.reshape(1,-1)
prediction=loaded_model.predict(input_array_reshaped)
print('Prediction of input data: ',prediction)
