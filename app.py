import numpy as np
import pickle
import streamlit as st

# 1. Load the saved model
loaded_model = pickle.load(open('heart_disease_model.sav', 'rb'))

# 2. Prediction Function
def heart_disease_prediction(input_data):
    input_data_as_numpy_array = np.asarray(input_data)
    input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)
    prediction = loaded_model.predict(input_data_reshaped)
    
    if prediction[0] == 0:
        return 'The person is Healthy. 🟢'
    else:
        return 'The person has Heart Disease. 🔴'

def main():
    
    st.title('Heart Disease Prediction System')
    st.write('Please select the patient\'s details from the dropdowns below:')

    col1, col2 = st.columns(2)
    
    with col1:
        # NUMERIC INPUTS (Keep as numbers)
        age = st.number_input('Age', min_value=1, max_value=120, value=25)
        trestbps = st.number_input('Resting Blood Pressure (mm Hg)', min_value=50, max_value=250, value=120)
        chol = st.number_input('Serum Cholesterol (mg/dl)', min_value=100, max_value=600, value=200)
        thalch = st.number_input('Maximum Heart Rate Achieved', min_value=50, max_value=250, value=150)
        oldpeak = st.number_input('Oldpeak (ST depression)', min_value=0.0, max_value=10.0, value=0.0)
        
    with col2:
        # CATEGORICAL INPUTS (Change to Dropdowns)
        
        # Sex (User sees Text -> Code converts to Number)
        sex_option = st.selectbox('Sex', ['Male', 'Female'])
        sex = 1 if sex_option == 'Male' else 0

        # Chest Pain Type
        cp_option = st.selectbox('Chest Pain Type', [
            'Typical Angina', 
            'Atypical Angina', 
            'Non-anginal Pain', 
            'Asymptomatic'
        ])
        # Mapping back to numbers based on your training
        if cp_option == 'Typical Angina': cp = 0
        elif cp_option == 'Atypical Angina': cp = 1
        elif cp_option == 'Non-anginal Pain': cp = 2
        else: cp = 3 # Asymptomatic

        # Fasting Blood Sugar
        fbs_option = st.selectbox('Fasting Blood Sugar > 120 mg/dl?', ['Yes', 'No'])
        fbs = 1 if fbs_option == 'Yes' else 0

        # Resting ECG
        restecg_option = st.selectbox('Resting ECG Results', [
            'Normal', 
            'ST-T Wave Abnormality', 
            'Left Ventricular Hypertrophy'
        ])
        if restecg_option == 'Normal': restecg = 0
        elif restecg_option == 'ST-T Wave Abnormality': restecg = 1
        else: restecg = 2

        # Exercise Induced Angina
        exang_option = st.selectbox('Exercise Induced Angina?', ['Yes', 'No'])
        exang = 1 if exang_option == 'Yes' else 0

    # Button
    if st.button('Get Test Result'):
        # Prepare the data for the model
        user_input = [age, sex, cp, trestbps, chol, fbs, restecg, thalch, exang, oldpeak]
        
        diagnosis = heart_disease_prediction(user_input)
        
        if "Healthy" in diagnosis:
            st.success(diagnosis)
        else:
            st.error(diagnosis)

if __name__ == '__main__':
    main()