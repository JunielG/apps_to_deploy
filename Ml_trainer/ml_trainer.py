import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.datasets import load_iris, load_wine, load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="ML Model Trainer",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-title {
        font-size: 3rem;
        color: #FF6B6B;
        text-align: center;
        margin-bottom: 2rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    .section-header {
        color: #4ECDC4;
        border-bottom: 2px solid #4ECDC4;
        padding-bottom: 10px;
        margin-top: 30px;
    }
    .model-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 15px;
        color: white;
        margin: 10px 0;
    }
    .metric-container {
        background-color: #f8f9fa;
        padding: 15px;
        border-radius: 10px;
        border-left: 4px solid #28a745;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)

# Title
st.markdown('<h1 class="main-title">🤖 Machine Learning Model Trainer</h1>', unsafe_allow_html=True)

# Sidebar - Dataset Selection
st.sidebar.header("🎯 Model Configuration")
st.sidebar.markdown("---")

# Dataset selection
dataset_name = st.sidebar.selectbox(
    "Choose Dataset",
    ["Iris Classification", "Wine Classification", "Breast Cancer Detection", "Custom Dataset"]
)

# Load dataset based on selection
@st.cache_data
def load_dataset(name):
    if name == "Iris Classification":
        data = load_iris()
        df = pd.DataFrame(data.data, columns=data.feature_names)
        df['target'] = data.target
        df['target_name'] = df['target'].map({i: name for i, name in enumerate(data.target_names)})
        return df, data.target_names
    elif name == "Wine Classification":
        data = load_wine()
        df = pd.DataFrame(data.data, columns=data.feature_names)
        df['target'] = data.target
        df['target_name'] = df['target'].map({i: name for i, name in enumerate(data.target_names)})
        return df, data.target_names
    elif name == "Breast Cancer Detection":
        data = load_breast_cancer()
        df = pd.DataFrame(data.data, columns=data.feature_names)
        df['target'] = data.target
        df['target_name'] = df['target'].map({i: name for i, name in enumerate(data.target_names)})
        return df, data.target_names
    else:
        return None, None

# Handle custom dataset upload
if dataset_name == "Custom Dataset":
    uploaded_file = st.sidebar.file_uploader("Upload CSV file", type=['csv'])
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.sidebar.success("Dataset uploaded successfully!")
        
        # Let user select target column
        target_column = st.sidebar.selectbox(
            "Select target column",
            df.columns.tolist()
        )
        
        if target_column:
            feature_columns = [col for col in df.columns if col != target_column]
            X = df[feature_columns]
            y = df[target_column]
            target_names = y.unique()
        else:
            st.stop()
    else:
        st.info("Please upload a CSV file to continue.")
        st.stop()
else:
    df, target_names = load_dataset(dataset_name)
    feature_columns = [col for col in df.columns if col not in ['target', 'target_name']]
    X = df[feature_columns]
    y = df['target']

# Model selection
st.sidebar.markdown("### 🎲 Model Selection")
model_type = st.sidebar.selectbox(
    "Choose Algorithm",
    ["Random Forest", "Logistic Regression", "Support Vector Machine"]
)

# Model parameters
st.sidebar.markdown("### ⚙️ Model Parameters")
if model_type == "Random Forest":
    n_estimators = st.sidebar.slider("Number of trees", 10, 200, 100)
    max_depth = st.sidebar.slider("Max depth", 1, 20, 10)
    model_params = {"n_estimators": n_estimators, "max_depth": max_depth, "random_state": 42}
elif model_type == "Logistic Regression":
    C = st.sidebar.slider("Regularization strength (C)", 0.01, 10.0, 1.0)
    model_params = {"C": C, "random_state": 42}
else:  # SVM
    C = st.sidebar.slider("Regularization strength (C)", 0.01, 10.0, 1.0)
    kernel = st.sidebar.selectbox("Kernel", ["rbf", "linear", "poly"])
    model_params = {"C": C, "kernel": kernel, "random_state": 42}

# Train-test split
test_size = st.sidebar.slider("Test size", 0.1, 0.5, 0.2)
random_state = st.sidebar.number_input("Random state", 0, 100, 42)

# Main content area
col1, col2 = st.columns([2, 1])

with col1:
    st.markdown('<h2 class="section-header">📊 Dataset Overview</h2>', unsafe_allow_html=True)
    
    # Dataset info
    st.write(f"**Dataset:** {dataset_name}")
    st.write(f"**Shape:** {df.shape}")
    st.write(f"**Features:** {len(feature_columns)}")
    st.write(f"**Classes:** {len(target_names)}")
    
    # Show first few rows
    st.subheader("Sample Data")
    st.dataframe(df.head(), use_container_width=True)

with col2:
    st.markdown('<h2 class="section-header">🎯 Target Distribution</h2>', unsafe_allow_html=True)
    
    # Target distribution
    if dataset_name != "Custom Dataset":
        target_counts = df['target_name'].value_counts()
        fig_pie = px.pie(
            values=target_counts.values,
            names=target_counts.index,
            title="Class Distribution"
        )
        st.plotly_chart(fig_pie, use_container_width=True)
    else:
        target_counts = y.value_counts()
        fig_pie = px.pie(
            values=target_counts.values,
            names=target_counts.index,
            title="Class Distribution"
        )
        st.plotly_chart(fig_pie, use_container_width=True)

# Feature analysis
st.markdown('<h2 class="section-header">🔍 Feature Analysis</h2>', unsafe_allow_html=True)

# Feature selection for analysis
selected_features = st.multiselect(
    "Select features for analysis:",
    feature_columns,
    default=feature_columns[:4] if len(feature_columns) >= 4 else feature_columns
)

if selected_features:
    # Correlation heatmap
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Feature Correlation")
        corr_matrix = df[selected_features].corr()
        fig_heatmap = px.imshow(
            corr_matrix,
            text_auto=True,
            aspect="auto",
            title="Feature Correlation Matrix"
        )
        st.plotly_chart(fig_heatmap, use_container_width=True)
    
    with col2:
        st.subheader("Feature Distribution")
        # Box plot for selected features
        if len(selected_features) > 0:
            feature_for_box = st.selectbox("Choose feature for box plot:", selected_features)
            if dataset_name != "Custom Dataset":
                fig_box = px.box(
                    df,
                    x='target_name',
                    y=feature_for_box,
                    title=f"{feature_for_box} by Class"
                )
            else:
                fig_box = px.box(
                    df,
                    x=target_column,
                    y=feature_for_box,
                    title=f"{feature_for_box} by Class"
                )
            st.plotly_chart(fig_box, use_container_width=True)

# Model training section
st.markdown('<h2 class="section-header">🚀 Model Training</h2>', unsafe_allow_html=True)

if st.button("🎯 Train Model", type="primary"):
    with st.spinner("Training model..."):
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Create and train model
        if model_type == "Random Forest":
            model = RandomForestClassifier(**model_params)
        elif model_type == "Logistic Regression":
            model = LogisticRegression(**model_params)
        else:
            model = SVC(**model_params)
        
        # Train model
        model.fit(X_train_scaled, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test_scaled)
        
        # Calculate accuracy
        accuracy = accuracy_score(y_test, y_pred)
        
        # Store results in session state
        st.session_state.model = model
        st.session_state.scaler = scaler
        st.session_state.accuracy = accuracy
        st.session_state.y_test = y_test
        st.session_state.y_pred = y_pred
        st.session_state.feature_columns = feature_columns
        st.session_state.target_names = target_names
        
        st.success(f"Model trained successfully! Accuracy: {accuracy:.2%}")

# Model results
if 'model' in st.session_state:
    st.markdown('<h2 class="section-header">📈 Model Performance</h2>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown(f"""
        <div class="metric-container">
            <h3>🎯 Accuracy</h3>
            <h2>{st.session_state.accuracy:.2%}</h2>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-container">
            <h3>📊 Test Samples</h3>
            <h2>{len(st.session_state.y_test)}</h2>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="metric-container">
            <h3>🔧 Model Type</h3>
            <h2>{model_type}</h2>
        </div>
        """, unsafe_allow_html=True)
    
    # Confusion Matrix
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Confusion Matrix")
        cm = confusion_matrix(st.session_state.y_test, st.session_state.y_pred)
        fig_cm = px.imshow(
            cm,
            text_auto=True,
            aspect="auto",
            title="Confusion Matrix",
            labels=dict(x="Predicted", y="Actual")
        )
        st.plotly_chart(fig_cm, use_container_width=True)
    
    with col2:
        st.subheader("Classification Report")
        report = classification_report(
            st.session_state.y_test, 
            st.session_state.y_pred, 
            target_names=[str(name) for name in st.session_state.target_names],
            output_dict=True
        )
        
        # Convert to DataFrame for better display
        report_df = pd.DataFrame(report).transpose()
        st.dataframe(report_df.round(3), use_container_width=True)

# Prediction section
if 'model' in st.session_state:
    st.markdown('<h2 class="section-header">🔮 Make Predictions</h2>', unsafe_allow_html=True)
    
    st.write("Enter feature values for prediction:")
    
    # Create input fields for each feature
    feature_inputs = {}
    cols = st.columns(min(3, len(st.session_state.feature_columns)))
    
    for i, feature in enumerate(st.session_state.feature_columns):
        col_idx = i % len(cols)
        with cols[col_idx]:
            # Get feature statistics for reasonable defaults
            feature_mean = X[feature].mean()
            feature_std = X[feature].std()
            feature_min = X[feature].min()
            feature_max = X[feature].max()
            
            feature_inputs[feature] = st.number_input(
                f"{feature}",
                min_value=float(feature_min),
                max_value=float(feature_max),
                value=float(feature_mean),
                step=float(feature_std/10)
            )
    
    if st.button("🎯 Predict", type="secondary"):
        # Prepare input data
        input_data = np.array([[feature_inputs[feature] for feature in st.session_state.feature_columns]])
        input_scaled = st.session_state.scaler.transform(input_data)
        
        # Make prediction
        prediction = st.session_state.model.predict(input_scaled)[0]
        
        # Get prediction probabilities if available
        if hasattr(st.session_state.model, 'predict_proba'):
            probabilities = st.session_state.model.predict_proba(input_scaled)[0]
            
            st.success(f"Prediction: {st.session_state.target_names[prediction]}")
            
            # Show probabilities
            st.subheader("Prediction Probabilities")
            prob_df = pd.DataFrame({
                'Class': st.session_state.target_names,
                'Probability': probabilities
            })
            
            fig_prob = px.bar(
                prob_df,
                x='Class',
                y='Probability',
                title='Prediction Probabilities'
            )
            st.plotly_chart(fig_prob, use_container_width=True)
        else:
            st.success(f"Prediction: {st.session_state.target_names[prediction]}")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; padding: 20px;'>
    <p>🤖 Machine Learning Model Trainer | Built with Streamlit & Scikit-learn</p>
    <p>Explore different algorithms and datasets to understand ML concepts!</p>
</div>
""", unsafe_allow_html=True)