import streamlit as st 
import numpy as np 
import joblib
import warnings 
import os
warnings.filterwarnings('ignore')

# Try loading with different protocols
def load_model_safely():
    try:
        with open('best_model.pkl', 'rb') as f:
            model = joblib.load(f)
        return model
    except Exception as e1:
        st.write(f"Standard pickle failed: {e1}")
        try:
            # Try with protocol 2 (older)
            with open('best_model.pkl', 'rb') as f:
                model = pickle.load(f, encoding='latin1')
            return model
        except Exception as e2:
            st.write(f"Latin1 encoding failed: {e2}")
            st.error("Could not load model with any method.")
            return None

model = load_model_safely()
if model is None:
    st.stop()

""" # Load the trained model with error handling
try:
    with open('best_model.pkl', 'rb') as f:
        model = pickle.load(f)     
except FileNotFoundError:
    st.error("Model file 'best_model.pkl' not found. Please ensure the file is in the correct directory.")
    st.stop()
except Exception as e:
    st.error(f"Error loading model: {str(e)}")
    st.stop()
""" 
if st.button('Predict Exam Score'):
    # Validate inputs
    if study_hours < 0 or study_hours > 12:
        st.error("Study hours should be between 0 and 12.")
    elif attendance < 0 or attendance > 100:
        st.error("Attendance should be between 0 and 100.")
    elif mental_health < 0 or mental_health > 10:
        st.error("Mental health should be between 0 and 10.")
    elif sleep_hours < 0 or sleep_hours > 12:
        st.error("Sleep hours should be between 0 and 12.")
    else:
        # Display result with interpretation
        st.success(f'Predicted Exam Score: {prediction:.2f}')

# Add performance interpretation
if prediction >= 90:
    st.balloons()
    st.write("🎉 Excellent performance!")
elif prediction >= 80:
    st.write("👍 Good performance!")
elif prediction >= 70:
    st.write("👌 Average performance!")
elif prediction >= 60:
    st.write("⚠️ Below average - consider improving study habits!")
else:
    st.write("🚨 Poor performance - significant improvement needed!")

# File Structure Check
# Make sure your file structure looks like this:
# your_project_folder/
#├── Streamlit_App.py
#└── best_model.pkl