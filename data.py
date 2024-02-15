import numpy as np
import pickle

from sklearn.preprocessing import scale
#loading the saved model
loaded_model=pickle.load(open(r'C:\Users\piyus\OneDrive\Desktop\New folder\trained_model.sav', 'rb'))
input_data= (1,89,66,23,94,28.1,0.167,21)
#changing the input data in nparray
input_datanp= np.asarray(input_data)
#reshape the array as we are predicting for one instance
input_data_reshape =input_datanp.reshape(1,-1)
#standarize the input data

prediction =loaded_model.predict(input_data_reshape)
print(prediction)
if (prediction == 0):
    print('not diabetic')
else:
    print('diabetic')