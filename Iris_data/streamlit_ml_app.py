import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score, classification_report
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="ML Model Deployment",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Title and description
st.title("🤖 Machine Learning Model Deployment")
st.markdown("### Interactive ML App with Streamlit")

# Sidebar for navigation
st.sidebar.header("Navigation")
page = st.sidebar.selectbox("Choose a page", ["Model Training", "Predictions", "Model Analysis"])

@st.cache_data
def load_data():
    """Load and prepare the dataset"""
    iris = load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    df['target'] = iris.target
    df['species'] = df['target'].map({0: 'setosa', 1: 'versicolor', 2: 'virginica'})
    return df, iris

@st.cache_resource
def train_model():
    """Train the machine learning model"""
    df, iris = load_data()
    X = df.drop(['target', 'species'], axis=1)
    y = df['target']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train Random Forest model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Calculate accuracy
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    return model, accuracy, X_test, y_test, y_pred, iris.feature_names, iris.target_names

# Load data and train model
df, iris_data = load_data()
model, accuracy, X_test, y_test, y_pred, feature_names, target_names = train_model()

if page == "Model Training":
    st.header("📊 Model Training & Dataset Overview")
    
    # Dataset overview
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Dataset Info")
        st.write(f"**Dataset Shape:** {df.shape}")
        st.write(f"**Features:** {len(feature_names)}")
        st.write(f"**Classes:** {len(target_names)}")
        st.write(f"**Model Accuracy:** {accuracy:.3f}")
        
        # Display dataset
        st.subheader("Dataset Preview")
        st.dataframe(df.head(10))
    
    with col2:
        st.subheader("Feature Distributions")
        feature_to_plot = st.selectbox("Select feature to visualize", feature_names)
        
        fig = px.histogram(df, x=feature_to_plot, color='species', 
                          title=f"Distribution of {feature_to_plot}")
        st.plotly_chart(fig, use_container_width=True)
    
    # Feature correlation heatmap
    st.subheader("Feature Correlation Matrix")
    corr_matrix = df[feature_names].corr()
    fig = px.imshow(corr_matrix, text_auto=True, aspect="auto",
                    title="Feature Correlation Heatmap")
    st.plotly_chart(fig, use_container_width=True)

elif page == "Predictions":
    st.header("🔮 Make Predictions")
    
    st.markdown("### Input Features")
    
    # Create input fields for predictions
    col1, col2 = st.columns(2)
    
    with col1:
        sepal_length = st.slider("Sepal Length (cm)", 
                                min_value=4.0, max_value=8.0, value=5.8, step=0.1)
        sepal_width = st.slider("Sepal Width (cm)", 
                               min_value=2.0, max_value=4.5, value=3.0, step=0.1)
    
    with col2:
        petal_length = st.slider("Petal Length (cm)", 
                                min_value=1.0, max_value=7.0, value=4.3, step=0.1)
        petal_width = st.slider("Petal Width (cm)", 
                               min_value=0.1, max_value=2.5, value=1.3, step=0.1)
    
    # Make prediction
    input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    prediction = model.predict(input_data)[0]
    prediction_proba = model.predict_proba(input_data)[0]
    
    st.markdown("### Prediction Results")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Predicted Class", target_names[prediction])
    
    with col2:
        st.metric("Confidence", f"{prediction_proba[prediction]:.3f}")
    
    with col3:
        st.metric("Model Accuracy", f"{accuracy:.3f}")
    
    # Prediction probabilities
    st.subheader("Prediction Probabilities")
    prob_df = pd.DataFrame({
        'Species': target_names,
        'Probability': prediction_proba
    })
    
    fig = px.bar(prob_df, x='Species', y='Probability', 
                 title="Prediction Probabilities for All Classes")
    st.plotly_chart(fig, use_container_width=True)
    
    # Input summary
    st.subheader("Input Summary")
    input_df = pd.DataFrame({
        'Feature': feature_names,
        'Value': [sepal_length, sepal_width, petal_length, petal_width]
    })
    st.dataframe(input_df, use_container_width=True)

elif page == "Model Analysis":
    st.header("📈 Model Analysis")
    
    # Model performance metrics
    from sklearn.metrics import confusion_matrix, classification_report
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Model Performance")
        st.write(f"**Accuracy:** {accuracy:.3f}")
        st.write(f"**Test Set Size:** {len(y_test)} samples")
        
        # Classification report
        report = classification_report(y_test, y_pred, target_names=target_names, output_dict=True)
        report_df = pd.DataFrame(report).iloc[:-1, :].T
        st.dataframe(report_df.round(3))
    
    with col2:
        st.subheader("Confusion Matrix")
        cm = confusion_matrix(y_test, y_pred)
        fig = px.imshow(cm, text_auto=True, aspect="auto",
                        labels=dict(x="Predicted", y="Actual"),
                        x=target_names, y=target_names,
                        title="Confusion Matrix")
        st.plotly_chart(fig, use_container_width=True)
    
    # Feature importance
    st.subheader("Feature Importance")
    importance = model.feature_importances_
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importance
    }).sort_values('Importance', ascending=False)
    
    fig = px.bar(importance_df, x='Importance', y='Feature', 
                 orientation='h', title="Feature Importance")
    st.plotly_chart(fig, use_container_width=True)
    
    # Model comparison (if you had multiple models)
    st.subheader("Model Information")
    st.write(f"**Model Type:** Random Forest Classifier")
    st.write(f"**Number of Estimators:** {model.n_estimators}")
    st.write(f"**Random State:** {model.random_state}")

# Sidebar additional info
st.sidebar.markdown("---")
st.sidebar.markdown("### About")
st.sidebar.info(
    "This app demonstrates machine learning model deployment using Streamlit. "
    "It includes model training, predictions, and analysis features."
)

st.sidebar.markdown("### Model Details")
st.sidebar.write(f"**Algorithm:** Random Forest")
st.sidebar.write(f"**Dataset:** Iris")
st.sidebar.write(f"**Accuracy:** {accuracy:.3f}")

# Footer
st.markdown("---")
st.markdown("Built with ❤️ using Streamlit")