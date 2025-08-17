import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Breast Cancer Prediction App",
    page_icon="🎗️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #FF69B4;
        text-align: center;
        padding: 1rem 0;
        border-bottom: 3px solid #FF69B4;
        margin-bottom: 2rem;
    }
    .metric-container {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
    }
    .prediction-result {
        font-size: 1.5rem;
        font-weight: bold;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        margin: 1rem 0;
    }
    .benign {
        background-color: #d4edda;
        color: #155724;
        border: 2px solid #c3e6cb;
    }
    .malignant {
        background-color: #f8d7da;
        color: #721c24;
        border: 2px solid #f5c6cb;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    """Load the breast cancer dataset"""
    try:
        df = pd.read_csv('breastcancer.csv')
        return df
    except FileNotFoundError:
        st.error("Please upload the breastcancer.csv file to use this app.")
        return None

def preprocess_data(df):
    """Preprocess the dataset"""
    # Remove ID column
    df = df.drop('id', axis=1)
    
    # Convert diagnosis to binary (M=1, B=0)
    df['diagnosis'] = df['diagnosis'].map({'M': 1, 'B': 0})
    
    return df

def train_models(X_train, X_test, y_train, y_test):
    """Train multiple models and return their performance"""
    models = {
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'SVM': SVC(random_state=42),
        'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000)
    }
    
    results = {}
    trained_models = {}
    
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        results[name] = {
            'accuracy': accuracy,
            'predictions': y_pred,
            'classification_report': classification_report(y_test, y_pred, output_dict=True)
        }
        trained_models[name] = model
    
    return results, trained_models

def main():
    st.markdown('<h1 class="main-header">🎗️ Breast Cancer Prediction App</h1>', unsafe_allow_html=True)
    
    # Load data
    df = load_data()
    if df is None:
        return
    
    # Sidebar for navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ["🏠 Home", "📊 Data Analysis", "🔮 Prediction", "📈 Model Performance"])
    
    if page == "🏠 Home":
        st.markdown("""
        ## Welcome to the Breast Cancer Prediction App
        
        This application uses machine learning to predict breast cancer diagnosis based on cell nucleus features.
        
        ### Dataset Information:
        - **Total Samples**: 569
        - **Features**: 30 numerical features computed from cell nucleus images
        - **Target**: Diagnosis (Malignant or Benign)
        
        ### Features Categories:
        1. **Mean values**: Average of measurements
        2. **Standard Error (SE)**: Standard error of measurements  
        3. **Worst values**: Mean of the three worst measurements
        
        ### Navigation:
        - **Data Analysis**: Explore the dataset with visualizations
        - **Prediction**: Make predictions using trained models
        - **Model Performance**: Compare different machine learning models
        
        ### Disclaimer:
        This app is for educational purposes only and should not be used for actual medical diagnosis.
        """)
        
        # Display dataset overview
        st.subheader("Dataset Overview")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total Samples", len(df))
        with col2:
            malignant_count = (df['diagnosis'] == 'M').sum()
            st.metric("Malignant Cases", malignant_count)
        with col3:
            benign_count = (df['diagnosis'] == 'B').sum()
            st.metric("Benign Cases", benign_count)
    
    elif page == "📊 Data Analysis":
        st.header("Data Analysis")
        
        # Preprocess data
        df_processed = preprocess_data(df.copy())
        
        # Diagnosis distribution
        st.subheader("Diagnosis Distribution")
        fig_dist = px.pie(df, names='diagnosis', title='Distribution of Diagnosis',
                         color_discrete_map={'M': '#ff6b6b', 'B': '#4ecdc4'})
        st.plotly_chart(fig_dist, use_container_width=True)
        
        # Feature correlation heatmap
        st.subheader("Feature Correlation Heatmap")
        
        # Select subset of features for better visualization
        mean_features = [col for col in df_processed.columns if col.endswith('_mean')]
        corr_matrix = df_processed[mean_features + ['diagnosis']].corr()
        
        fig_corr = px.imshow(corr_matrix, 
                           title='Correlation Matrix of Mean Features',
                           color_continuous_scale='RdBu_r',
                           aspect='auto')
        st.plotly_chart(fig_corr, use_container_width=True)
        
        # Feature distributions
        st.subheader("Feature Distributions by Diagnosis")
        
        feature_to_plot = st.selectbox("Select feature to plot:", mean_features)
        
        fig_violin = px.violin(df, x='diagnosis', y=feature_to_plot, 
                              title=f'Distribution of {feature_to_plot} by Diagnosis',
                              color='diagnosis',
                              color_discrete_map={'M': '#ff6b6b', 'B': '#4ecdc4'})
        st.plotly_chart(fig_violin, use_container_width=True)
        
        # PCA visualization
        st.subheader("PCA Visualization")
        
        X = df_processed.drop('diagnosis', axis=1)
        y = df_processed['diagnosis']
        
        # Standardize features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Apply PCA
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X_scaled)
        
        pca_df = pd.DataFrame({
            'PC1': X_pca[:, 0],
            'PC2': X_pca[:, 1],
            'Diagnosis': ['Malignant' if d == 1 else 'Benign' for d in y]
        })
        
        fig_pca = px.scatter(pca_df, x='PC1', y='PC2', color='Diagnosis',
                           title='PCA Visualization of Breast Cancer Data',
                           color_discrete_map={'Malignant': '#ff6b6b', 'Benign': '#4ecdc4'})
        st.plotly_chart(fig_pca, use_container_width=True)
        
        st.info(f"PC1 explains {pca.explained_variance_ratio_[0]:.2%} of variance")
        st.info(f"PC2 explains {pca.explained_variance_ratio_[1]:.2%} of variance")
    
    elif page == "🔮 Prediction":
        st.header("Breast Cancer Prediction")
        
        # Preprocess data
        df_processed = preprocess_data(df.copy())
        
        # Split features and target
        X = df_processed.drop('diagnosis', axis=1)
        y = df_processed['diagnosis']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train models
        results, trained_models = train_models(X_train_scaled, X_test_scaled, y_train, y_test)
        
        # Model selection
        st.subheader("Select Model for Prediction")
        selected_model = st.selectbox("Choose a model:", list(trained_models.keys()))
        
        # Feature input
        st.subheader("Input Features for Prediction")
        
        # Create input fields for all features
        input_features = {}
        
        # Group features by category
        mean_features = [col for col in X.columns if col.endswith('_mean')]
        se_features = [col for col in X.columns if col.endswith('_se')]
        worst_features = [col for col in X.columns if col.endswith('_worst')]
        
        tab1, tab2, tab3 = st.tabs(["Mean Features", "Standard Error Features", "Worst Features"])
        
        with tab1:
            col1, col2 = st.columns(2)
            for i, feature in enumerate(mean_features):
                col = col1 if i % 2 == 0 else col2
                with col:
                    min_val = float(X[feature].min())
                    max_val = float(X[feature].max())
                    mean_val = float(X[feature].mean())
                    input_features[feature] = st.number_input(
                        f"{feature.replace('_mean', '').replace('_', ' ').title()}",
                        min_value=min_val,
                        max_value=max_val,
                        value=mean_val,
                        key=f"mean_{feature}"
                    )
        
        with tab2:
            col1, col2 = st.columns(2)
            for i, feature in enumerate(se_features):
                col = col1 if i % 2 == 0 else col2
                with col:
                    min_val = float(X[feature].min())
                    max_val = float(X[feature].max())
                    mean_val = float(X[feature].mean())
                    input_features[feature] = st.number_input(
                        f"{feature.replace('_se', '').replace('_', ' ').title()} (SE)",
                        min_value=min_val,
                        max_value=max_val,
                        value=mean_val,
                        key=f"se_{feature}"
                    )
        
        with tab3:
            col1, col2 = st.columns(2)
            for i, feature in enumerate(worst_features):
                col = col1 if i % 2 == 0 else col2
                with col:
                    min_val = float(X[feature].min())
                    max_val = float(X[feature].max())
                    mean_val = float(X[feature].mean())
                    input_features[feature] = st.number_input(
                        f"{feature.replace('_worst', '').replace('_', ' ').title()} (Worst)",
                        min_value=min_val,
                        max_value=max_val,
                        value=mean_val,
                        key=f"worst_{feature}"
                    )
        
        # Make prediction
        if st.button("Make Prediction", type="primary"):
            # Prepare input data
            input_data = np.array([list(input_features.values())])
            input_data_scaled = scaler.transform(input_data)
            
            # Get prediction
            model = trained_models[selected_model]
            prediction = model.predict(input_data_scaled)[0]
            prediction_proba = model.predict_proba(input_data_scaled)[0]
            
            # Display result
            if prediction == 1:
                st.markdown(
                    '<div class="prediction-result malignant">⚠️ Malignant (Cancerous)</div>',
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    '<div class="prediction-result benign">✅ Benign (Non-cancerous)</div>',
                    unsafe_allow_html=True
                )
            
            # Show confidence
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Benign Probability", f"{prediction_proba[0]:.2%}")
            with col2:
                st.metric("Malignant Probability", f"{prediction_proba[1]:.2%}")
            
            # Show model accuracy
            st.info(f"Model Accuracy: {results[selected_model]['accuracy']:.2%}")
    
    elif page == "📈 Model Performance":
        st.header("Model Performance Comparison")
        
        # Preprocess data
        df_processed = preprocess_data(df.copy())
        
        # Split features and target
        X = df_processed.drop('diagnosis', axis=1)
        y = df_processed['diagnosis']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train models
        results, trained_models = train_models(X_train_scaled, X_test_scaled, y_train, y_test)
        
        # Performance comparison
        st.subheader("Model Accuracy Comparison")
        
        accuracies = {model: results[model]['accuracy'] for model in results}
        
        fig_acc = px.bar(x=list(accuracies.keys()), y=list(accuracies.values()),
                        title='Model Accuracy Comparison',
                        labels={'x': 'Models', 'y': 'Accuracy'},
                        color=list(accuracies.values()),
                        color_continuous_scale='viridis')
        st.plotly_chart(fig_acc, use_container_width=True)
        
        # Detailed metrics
        st.subheader("Detailed Model Metrics")
        
        selected_model_perf = st.selectbox("Select model for detailed metrics:", list(results.keys()))
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Classification Report")
            report = results[selected_model_perf]['classification_report']
            
            # Create metrics DataFrame
            metrics_df = pd.DataFrame({
                'Class': ['Benign (0)', 'Malignant (1)', 'Accuracy', 'Macro Avg', 'Weighted Avg'],
                'Precision': [
                    report['0']['precision'],
                    report['1']['precision'],
                    report['accuracy'],
                    report['macro avg']['precision'],
                    report['weighted avg']['precision']
                ],
                'Recall': [
                    report['0']['recall'],
                    report['1']['recall'],
                    report['accuracy'],
                    report['macro avg']['recall'],
                    report['weighted avg']['recall']
                ],
                'F1-Score': [
                    report['0']['f1-score'],
                    report['1']['f1-score'],
                    report['accuracy'],
                    report['macro avg']['f1-score'],
                    report['weighted avg']['f1-score']
                ]
            })
            
            st.dataframe(metrics_df, use_container_width=True)
        
        with col2:
            st.subheader("Confusion Matrix")
            predictions = results[selected_model_perf]['predictions']
            cm = confusion_matrix(y_test, predictions)
            
            fig_cm = px.imshow(cm, 
                             labels=dict(x="Predicted", y="Actual", color="Count"),
                             x=['Benign', 'Malignant'],
                             y=['Benign', 'Malignant'],
                             title='Confusion Matrix',
                             color_continuous_scale='Blues')
            
            # Add text annotations
            for i in range(len(cm)):
                for j in range(len(cm[0])):
                    fig_cm.add_annotation(
                        x=j, y=i,
                        text=str(cm[i][j]),
                        showarrow=False,
                        font=dict(color="white" if cm[i][j] > cm.max()/2 else "black")
                    )
            
            st.plotly_chart(fig_cm, use_container_width=True)

if __name__ == "__main__":
    main()