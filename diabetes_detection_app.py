import numpy as np
import pickle
import streamlit as st

# Load the model
loaded_model = pickle.load(open(r'C:\Users\piyus\OneDrive\Desktop\New folder\trained_model.sav', 'rb'))

# Creating a function for prediction
def diabetes_prediction(input_data):
    # Change the input data to a numpy array
    input_data_np = np.asarray(input_data)
    # Reshape the array as we are predicting for one instance
    input_data_reshaped = input_data_np.reshape(1, -1)
    # Standardize the input data (if your model requires standardization, you need to apply the same scaler here)

    prediction = loaded_model.predict(input_data_reshaped)
    print(prediction)
    if prediction == 0:
        return 'Not diabetic'
    else:
        return 'Diabetic'

def main():
    st.title('Diabetes Prediction Web App')
    # Getting input data from user
    pregnancies = st.text_input('Number of Pregnancies')
    glucose = st.text_input('Glucose Level')
    blood_pressure = st.text_input('Blood Pressure Level')
    skin_thickness = st.text_input('Skin Thickness Level')
    insulin = st.text_input('Insulin Level')
    bmi = st.text_input('BMI Level')
    diabetes_pedigree_function = st.text_input('Diabetes Pedigree Function')
    age = st.text_input('Age')
    
    # Code for prediction
    diagnosis = ''
    # Creating a button for prediction
    if st.button('Diabetes Test Result'):
        diagnosis = diabetes_prediction([
            float(pregnancies), 
            float(glucose), 
            float(blood_pressure), 
            float(skin_thickness), 
            float(insulin), 
            float(bmi), 
            float(diabetes_pedigree_function), 
            float(age)
        ])

    st.success(diagnosis)

if __name__ == '__main__':
    main()