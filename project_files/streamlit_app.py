import streamlit as st 
import numpy as np 
import joblib
import warnings 
warnings.filterwarnings('ignore')

# Load the trained model
model = joblib.load('best_model.pkl')

st.title('Student Exam Score Predictor')

# Create input sliders
study_hours = st.slider('Study Hours Per Day', 0.0, 12.0, 2.0)
attendance = st.slider('Attendance Percentage', 0.0, 100.0, 80.0)
mental_health = st.slider('Mental Health Rating (1-10)', 1, 10, 5)
sleep_hours = st.slider('Sleep Hours Per Night', 0.0, 12.0, 7.0)

# Fixed: selectbox instead of slider for categorical data
part_time_job = st.selectbox('Part-Time Job', ['No', 'Yes'])

# Encode the categorical variable
ptj_encoded = 1 if part_time_job == 'Yes' else 0

# Prediction button
if st.button('Predict Exam Score'):
    # Prepare input data
    input_data = np.array([[study_hours, attendance, mental_health, sleep_hours, ptj_encoded]])
    
    # Make prediction
    prediction = model.predict(input_data)[0]
    
    # Ensure prediction is within valid range (0-100)
    prediction = max(0, min(100, prediction))
    
    # Display result
    st.success(f'Predicted Exam Score: {prediction:.2f}')
    
    # Optional: Add some additional context
    st.info(f"""
    **Input Summary:**
    - Study Hours: {study_hours} hours/day
    - Attendance: {attendance}%
    - Mental Health: {mental_health}/10
    - Sleep Hours: {sleep_hours} hours/night
    - Part-Time Job: {part_time_job}
    """)
