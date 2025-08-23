import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Set page config
st.set_page_config(
    page_title="Diabetes Prediction App",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-container {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    """Load the diabetes dataset"""
    # You would replace this with your actual data loading
    # For demo purposes, I'm creating sample data matching your CSV structure
    np.random.seed(42)
    n_samples = 768
    
    data = {
        'Pregnancies': np.random.randint(0, 17, n_samples),
        'Glucose': np.random.randint(0, 200, n_samples),
        'BloodPressure': np.random.randint(0, 122, n_samples),
        'SkinThickness': np.random.randint(0, 99, n_samples),
        'Insulin': np.random.randint(0, 846, n_samples),
        'BMI': np.random.uniform(0, 67.1, n_samples),
        'DiabetesPedigreeFunction': np.random.uniform(0.078, 2.42, n_samples),
        'Age': np.random.randint(21, 81, n_samples),
        'Outcome': np.random.choice([0, 1], n_samples, p=[0.65, 0.35])
    }
    
    return pd.DataFrame(data)

# Load data
df = load_data()

# Sidebar
st.sidebar.title("🩺 Diabetes Prediction")
st.sidebar.markdown("---")

# Navigation
page = st.sidebar.selectbox(
    "Select Page",
    ["📊 Data Overview", "📈 Data Visualization", "🤖 Model Training", "🔮 Make Predictions"]
)

# Main content
st.markdown('<h1 class="main-header">Diabetes Prediction Application</h1>', unsafe_allow_html=True)

if page == "📊 Data Overview":
    st.header("Dataset Overview")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Records", len(df))
    with col2:
        st.metric("Features", len(df.columns) - 1)
    with col3:
        st.metric("Diabetes Cases", df['Outcome'].sum())
    with col4:
        st.metric("Non-Diabetes Cases", len(df) - df['Outcome'].sum())
    
    st.subheader("Dataset Sample")
    st.dataframe(df.head(10))
    
    st.subheader("Statistical Summary")
    st.dataframe(df.describe())
    
    st.subheader("Missing Values")
    missing_data = df.isnull().sum()
    if missing_data.sum() == 0:
        st.success("No missing values found in the dataset!")
    else:
        st.dataframe(missing_data)
    
    st.subheader("Data Types")
    st.dataframe(df.dtypes)
    
    st.subheader("Feature Descriptions")
    feature_descriptions = {
        'Pregnancies': 'Number of times pregnant',
        'Glucose': 'Plasma glucose concentration',
        'BloodPressure': 'Diastolic blood pressure (mm Hg)',
        'SkinThickness': 'Triceps skin fold thickness (mm)',
        'Insulin': '2-Hour serum insulin (mu U/ml)',
        'BMI': 'Body mass index (weight in kg/(height in m)^2)',
        'DiabetesPedigreeFunction': 'Diabetes pedigree function',
        'Age': 'Age (years)',
        'Outcome': 'Class variable (0 or 1) - 1 indicates diabetes'
    }
    
    for feature, description in feature_descriptions.items():
        st.write(f"**{feature}**: {description}")

elif page == "📈 Data Visualization":
    st.header("Data Visualization")
    
    # Outcome distribution
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Diabetes Outcome Distribution")
        outcome_counts = df['Outcome'].value_counts()
        fig_pie = px.pie(
            values=outcome_counts.values,
            names=['Non-Diabetic', 'Diabetic'],
            title="Distribution of Diabetes Cases"
        )
        st.plotly_chart(fig_pie, use_container_width=True)
    
    with col2:
        st.subheader("Age Distribution by Outcome")
        fig_age = px.histogram(
            df, x='Age', color='Outcome',
            title="Age Distribution by Diabetes Outcome",
            nbins=30
        )
        st.plotly_chart(fig_age, use_container_width=True)
    
    # Feature distributions
    st.subheader("Feature Distributions")
    features = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 
                'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
    
    selected_features = st.multiselect(
        "Select features to visualize:",
        features,
        default=['Glucose', 'BMI', 'Age']
    )
    
    if selected_features:
        fig_dist = make_subplots(
            rows=len(selected_features), cols=1,
            subplot_titles=selected_features
        )
        
        for i, feature in enumerate(selected_features):
            for outcome in [0, 1]:
                data = df[df['Outcome'] == outcome][feature]
                fig_dist.add_trace(
                    go.Histogram(
                        x=data,
                        name=f"{'Diabetic' if outcome == 1 else 'Non-Diabetic'}",
                        opacity=0.7
                    ),
                    row=i+1, col=1
                )
        
        fig_dist.update_layout(height=300*len(selected_features))
        st.plotly_chart(fig_dist, use_container_width=True)
    
    # Correlation matrix
    st.subheader("Feature Correlation Matrix")
    correlation_matrix = df.corr()
    fig_corr = px.imshow(
        correlation_matrix,
        title="Feature Correlation Matrix",
        color_continuous_scale='RdBu'
    )
    st.plotly_chart(fig_corr, use_container_width=True)
    
    # Box plots
    st.subheader("Feature Box Plots by Outcome")
    selected_box_feature = st.selectbox(
        "Select feature for box plot:",
        features
    )
    
    fig_box = px.box(
        df, x='Outcome', y=selected_box_feature,
        title=f"{selected_box_feature} Distribution by Diabetes Outcome"
    )
    st.plotly_chart(fig_box, use_container_width=True)

elif page == "🤖 Model Training":
    st.header("Machine Learning Model Training")
    
    # Model selection
    st.subheader("Model Configuration")
    col1, col2 = st.columns(2)
    
    with col1:
        model_type = st.selectbox(
            "Select Model Type:",
            ["Random Forest", "Logistic Regression", "Support Vector Machine"]
        )
    
    with col2:
        test_size = st.slider("Test Set Size:", 0.1, 0.5, 0.2, 0.05)
    
    # Feature selection
    st.subheader("Feature Selection")
    features = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 
                'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
    
    selected_features = st.multiselect(
        "Select features for training:",
        features,
        default=features
    )
    
    if st.button("Train Model"):
        if len(selected_features) == 0:
            st.error("Please select at least one feature!")
        else:
            # Prepare data
            X = df[selected_features]
            y = df['Outcome']
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=42
            )
            
            # Scale features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # Train model
            with st.spinner("Training model..."):
                if model_type == "Random Forest":
                    model = RandomForestClassifier(n_estimators=100, random_state=42)
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)
                elif model_type == "Logistic Regression":
                    model = LogisticRegression(random_state=42)
                    model.fit(X_train_scaled, y_train)
                    y_pred = model.predict(X_test_scaled)
                else:  # SVM
                    model = SVC(random_state=42)
                    model.fit(X_train_scaled, y_train)
                    y_pred = model.predict(X_test_scaled)
            
            # Display results
            accuracy = accuracy_score(y_test, y_pred)
            
            st.success(f"Model trained successfully!")
            st.metric("Accuracy", f"{accuracy:.3f}")
            
            # Confusion matrix
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Confusion Matrix")
                cm = confusion_matrix(y_test, y_pred)
                fig_cm = px.imshow(
                    cm,
                    labels=dict(x="Predicted", y="Actual"),
                    x=['Non-Diabetic', 'Diabetic'],
                    y=['Non-Diabetic', 'Diabetic'],
                    title="Confusion Matrix"
                )
                st.plotly_chart(fig_cm, use_container_width=True)
            
            with col2:
                st.subheader("Classification Report")
                report = classification_report(y_test, y_pred, output_dict=True)
                st.json(report)
            
            # Feature importance (for Random Forest)
            if model_type == "Random Forest":
                st.subheader("Feature Importance")
                importance_df = pd.DataFrame({
                    'Feature': selected_features,
                    'Importance': model.feature_importances_
                }).sort_values('Importance', ascending=False)
                
                fig_importance = px.bar(
                    importance_df,
                    x='Importance',
                    y='Feature',
                    orientation='h',
                    title="Feature Importance"
                )
                st.plotly_chart(fig_importance, use_container_width=True)
            
            # Store model in session state
            st.session_state['model'] = model
            st.session_state['scaler'] = scaler
            st.session_state['selected_features'] = selected_features
            st.session_state['model_type'] = model_type

elif page == "🔮 Make Predictions":
    st.header("Make Diabetes Predictions")
    
    if 'model' not in st.session_state:
        st.warning("Please train a model first!")
        st.info("Go to the 'Model Training' page to train a model.")
    else:
        st.subheader("Enter Patient Information")
        
        # Create input fields
        col1, col2 = st.columns(2)
        
        with col1:
            pregnancies = st.number_input("Pregnancies", min_value=0, max_value=20, value=0)
            glucose = st.number_input("Glucose", min_value=0, max_value=200, value=100)
            blood_pressure = st.number_input("Blood Pressure", min_value=0, max_value=150, value=70)
            skin_thickness = st.number_input("Skin Thickness", min_value=0, max_value=100, value=20)
        
        with col2:
            insulin = st.number_input("Insulin", min_value=0, max_value=900, value=30)
            bmi = st.number_input("BMI", min_value=0.0, max_value=70.0, value=25.0)
            diabetes_pedigree = st.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=3.0, value=0.5)
            age = st.number_input("Age", min_value=18, max_value=120, value=30)
        
        if st.button("Predict Diabetes Risk"):
            # Prepare input data
            input_data = {
                'Pregnancies': pregnancies,
                'Glucose': glucose,
                'BloodPressure': blood_pressure,
                'SkinThickness': skin_thickness,
                'Insulin': insulin,
                'BMI': bmi,
                'DiabetesPedigreeFunction': diabetes_pedigree,
                'Age': age
            }
            
            # Select only the features used in training
            input_df = pd.DataFrame([input_data])
            input_features = input_df[st.session_state['selected_features']]
            
            # Scale if necessary
            if st.session_state['model_type'] in ['Logistic Regression', 'Support Vector Machine']:
                input_features = st.session_state['scaler'].transform(input_features)
            
            # Make prediction
            prediction = st.session_state['model'].predict(input_features)[0]
            prediction_proba = st.session_state['model'].predict_proba(input_features)[0]
            
            # Display results
            st.subheader("Prediction Results")
            
            if prediction == 1:
                st.error("⚠️ High Risk of Diabetes")
                st.write(f"Probability of Diabetes: {prediction_proba[1]:.2%}")
            else:
                st.success("✅ Low Risk of Diabetes")
                st.write(f"Probability of No Diabetes: {prediction_proba[0]:.2%}")
            
            # Risk gauge
            fig_gauge = go.Figure(go.Indicator(
                mode = "gauge+number",
                value = prediction_proba[1] * 100,
                domain = {'x': [0, 1], 'y': [0, 1]},
                title = {'text': "Diabetes Risk (%)"},
                gauge = {
                    'axis': {'range': [None, 100]},
                    'bar': {'color': "darkblue"},
                    'steps': [
                        {'range': [0, 50], 'color': "lightgray"},
                        {'range': [50, 100], 'color': "gray"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 50
                    }
                }
            ))
            st.plotly_chart(fig_gauge, use_container_width=True)
            
            # Recommendations
            st.subheader("Recommendations")
            if prediction == 1:
                st.markdown("""
                **🏥 Please consult with a healthcare provider for proper evaluation and testing.**
                
                **General recommendations:**
                - Monitor blood glucose levels regularly
                - Maintain a healthy diet low in sugar and refined carbohydrates
                - Exercise regularly (at least 150 minutes per week)
                - Maintain a healthy weight
                - Get regular check-ups
                """)
            else:
                st.markdown("""
                **✅ Continue maintaining a healthy lifestyle:**
                
                - Eat a balanced diet rich in vegetables and whole grains
                - Stay physically active
                - Maintain a healthy weight
                - Get regular health screenings
                - Limit processed foods and sugary drinks
                """)

# Footer
st.markdown("---")
st.markdown("🩺 **Diabetes Prediction App** | Built with Streamlit")
st.markdown("⚠️ **Disclaimer:** This app is for educational purposes only. Always consult healthcare professionals for medical advice.")