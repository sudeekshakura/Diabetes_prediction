#Importing the Dependencies

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import svm 
from sklearn.metrics import accuracy_score

#Data Collection and Analysis
#loading dataset diabetes to a pandas dataset
diabetes_dataset=pd.read_csv('diabetes.csv')
diabetes_dataset.head(5)

#number of rows and columns in diabetes df
diabetes_dataset.shape

#Getting the statistical measures of the data
diabetes_dataset.describe()

diabetes_dataset['Outcome'].value_counts()

diabetes_dataset.groupby('Outcome').mean()

# Separarting Data and Labels
X=diabetes_dataset.drop(columns='Outcome',axis=1)
Y=diabetes_dataset['Outcome']

#Data Standardization
#Each column is in different range so hence we need to standardize the data

X=X
Y=diabetes_dataset['Outcome']

print(X)
print(Y)
# Train Test Split
X_train, X_test, Y_train, Y_test=train_test_split(X,Y,test_size=0.2, stratify=Y, random_state=2)
print(X.shape, X_train.shape, X_test.shape)

# Train the model

classifier=svm.SVC(kernel='linear')

#training the support vector machine classifier 
classifier.fit(X_train, Y_train)

#Evaluate Model
#Accuracy Score on training data
X_train_prediction=classifier.predict(X_train)
training_data_accuracy=accuracy_score(X_train_prediction,Y_train)
print('Accuracy Score of training data: ',training_data_accuracy)


#Accuracy Score on test data
X_test_prediction=classifier.predict(X_test)
test_data_accuracy=accuracy_score(X_test_prediction,Y_test)
print('Accuracy Score of test data: ',test_data_accuracy)

#Making a predictive system 

input_data=(4,110,92,0,0,37.6,0.191,30)

#Change input data to numpy array
input_data_array=np.asarray(input_data)

#reshape the numpy array
input_array_reshaped= input_data_array.reshape(1,-1)
prediction=classifier.predict(input_array_reshaped)
print('Prediction of input data: ',prediction)

if prediction[0]==0:
    print('Patient is non-diabetic.')
else:
    print('Patient is diabetic.')

#Saving the trained Model

import pickle 

filename='trained_model.sav'

#wb --> Write Binary
pickle.dump(classifier,open(filename, 'wb'))

#Loading the saved model
#rb-->reading the binary
# if using standardization then include the standardization in pickle too 
loaded_model= pickle.load(open('trained_model.sav','rb'))

#Making a predictive system 

input_data=(4,110,92,0,0,37.6,0.191,30)

#Change input data to numpy array
input_data_array=np.asarray(input_data)

#reshape the numpy array
input_array_reshaped= input_data_array.reshape(1,-1)
prediction=loaded_model.predict(input_array_reshaped)
print('Prediction of input data: ',prediction)
