import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import matplotlib.pyplot as plt

# Set page config
st.set_page_config(
    page_title="Loan Approval Prediction System",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main {
        padding-top: 1rem;
    }
    .stMetric {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .prediction-box {
        background-color: #f8f9fa;
        padding: 1.5rem;
        border-radius: 0.5rem;
        border: 1px solid #dee2e6;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Title and description
st.title("🏦 Loan Approval Prediction System")
st.markdown("---")

# Sidebar for navigation
st.sidebar.title("Navigation")
page = st.sidebar.selectbox("Choose a page", ["Data Overview", "Model Training", "Loan Prediction", "Model Performance"])

# Load and cache data
@st.cache_data
def load_data():
    """Load the loan data from CSV file"""
    try:
        data = pd.read_csv('loans.csv')
        return data
    except FileNotFoundError:
        st.error("CSV file not found. Please ensure 'loans.csv' is in the same directory as this app.")
        return None

# Load data
df = load_data()

if df is not None:
    # Data preprocessing
    @st.cache_data
    def preprocess_data(data):
        """Preprocess the data for modeling"""
        # Convert Default to binary (Yes=1, No=0)
        data['Default_Binary'] = (data['Default'] == 'Yes').astype(int)
        
        # Create feature columns
        feature_columns = ['Age', 'Annual_Income', 'Credit_Score', 'Employment_Years', 'Loan_Amount_Requested']
        X = data[feature_columns]
        y = data['Default_Binary']
        
        return X, y, feature_columns
    
    X, y, feature_columns = preprocess_data(df)
    
    # Page 1: Data Overview
    if page == "Data Overview":
        st.header("📊 Data Overview")
        
        # Basic statistics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Applications", len(df))
        with col2:
            st.metric("Approved Loans", len(df[df['Default'] == 'No']))
        with col3:
            st.metric("Rejected Loans", len(df[df['Default'] == 'Yes']))
        with col4:
            approval_rate = (len(df[df['Default'] == 'No']) / len(df)) * 100
            st.metric("Approval Rate", f"{approval_rate:.1f}%")
        
        # Display data sample
        st.subheader("Sample Data")
        st.dataframe(df.head(10))
        
        # Data distribution visualizations
        st.subheader("Data Distribution")
        
        # Create tabs for different visualizations
        tab1, tab2, tab3 = st.tabs(["Feature Distributions", "Correlation Matrix", "Default Analysis"])
        
        with tab1:
            # Feature distributions
            fig_dist = make_subplots(
                rows=2, cols=3,
                subplot_titles=('Age', 'Annual Income', 'Credit Score', 'Employment Years', 'Loan Amount', 'Default Status'),
                specs=[[{"type": "histogram"}, {"type": "histogram"}, {"type": "histogram"}],
                       [{"type": "histogram"}, {"type": "histogram"}, {"type": "pie"}]]
            )
            
            # Add histograms
            fig_dist.add_trace(go.Histogram(x=df['Age'], name='Age'), row=1, col=1)
            fig_dist.add_trace(go.Histogram(x=df['Annual_Income'], name='Annual Income'), row=1, col=2)
            fig_dist.add_trace(go.Histogram(x=df['Credit_Score'], name='Credit Score'), row=1, col=3)
            fig_dist.add_trace(go.Histogram(x=df['Employment_Years'], name='Employment Years'), row=2, col=1)
            fig_dist.add_trace(go.Histogram(x=df['Loan_Amount_Requested'], name='Loan Amount'), row=2, col=2)
            
            # Add pie chart for default status
            default_counts = df['Default'].value_counts()
            fig_dist.add_trace(go.Pie(labels=default_counts.index, values=default_counts.values, name='Default Status'), row=2, col=3)
            
            fig_dist.update_layout(height=600, showlegend=False, title_text="Feature Distributions")
            st.plotly_chart(fig_dist, use_container_width=True)
        
        with tab2:
            # Correlation matrix
            numeric_df = df.select_dtypes(include=[np.number])
            correlation_matrix = numeric_df.corr()
            
            fig_corr = px.imshow(correlation_matrix, 
                               text_auto=True, 
                               aspect="auto",
                               title="Feature Correlation Matrix")
            st.plotly_chart(fig_corr, use_container_width=True)
        
        with tab3:
            # Default analysis by different factors
            col1, col2 = st.columns(2)
            
            with col1:
                # Default rate by credit score bins
                df['Credit_Score_Bin'] = pd.cut(df['Credit_Score'], bins=5, labels=['Very Low', 'Low', 'Medium', 'High', 'Very High'])
                
                default_by_credit = df.groupby('Credit_Score_Bin')['Default'].apply(lambda x: (x == 'Yes').mean() * 100).reset_index()
                
                default_by_credit.columns = ['Credit_Score_Bin', 'Default_Rate']
                
                fig_credit = px.bar(default_by_credit, x='Credit_Score_Bin', y='Default_Rate',
                                  title="Default Rate by Credit Score Range")
                st.plotly_chart(fig_credit, use_container_width=True)
            
            with col2:
                # Default rate by income bins
                df['Income_Bin'] = pd.cut(df['Annual_Income'], bins=5, labels=['Very Low', 'Low', 'Medium', 'High', 'Very High'])
                default_by_income = df.groupby('Income_Bin')['Default'].apply(lambda x: (x == 'Yes').mean() * 100).reset_index()
               
                default_by_income.columns = ['Income_Bin', 'Default_Rate']
                
                fig_income = px.bar(default_by_income, x='Income_Bin', y='Default_Rate',
                                  title="Default Rate by Income Range")
                st.plotly_chart(fig_income, use_container_width=True)
    
    # Page 2: Model Training
    elif page == "Model Training":
        st.header("🤖 Model Training")
        
        # Model selection
        st.subheader("Model Configuration")
        col1, col2 = st.columns(2)
        
        with col1:
            model_type = st.selectbox("Select Model Type", ["Random Forest", "Logistic Regression"])
            test_size = st.slider("Test Size", 0.1, 0.5, 0.2, 0.05)
        
        with col2:
            random_state = st.number_input("Random State", value=42)
            if model_type == "Random Forest":
                n_estimators = st.slider("Number of Estimators", 10, 200, 100, 10)
        
        # Train model button
        if st.button("Train Model", type="primary"):
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
            
            # Scale features for logistic regression
            if model_type == "Logistic Regression":
                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train)
                X_test_scaled = scaler.transform(X_test)
                
                # Train logistic regression
                model = LogisticRegression(random_state=random_state)
                model.fit(X_train_scaled, y_train)
                y_pred = model.predict(X_test_scaled)
                
                # Store scaler in session state
                st.session_state['scaler'] = scaler
            else:
                # Train random forest
                model = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state)
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
            
            # Store model in session state
            st.session_state['model'] = model
            st.session_state['model_type'] = model_type
            st.session_state['feature_columns'] = feature_columns
            
            # Calculate metrics
            accuracy = accuracy_score(y_test, y_pred)
            
            # Display results
            st.success(f"Model trained successfully! Accuracy: {accuracy:.3f}")
            
            # Confusion matrix
            cm = confusion_matrix(y_test, y_pred)
            fig_cm = px.imshow(cm, 
                             text_auto=True, 
                             aspect="auto",
                             title="Confusion Matrix",
                             labels=dict(x="Predicted", y="Actual"))
            st.plotly_chart(fig_cm, use_container_width=True)
            
            # Classification report
            st.subheader("Classification Report")
            report = classification_report(y_test, y_pred, output_dict=True)
            report_df = pd.DataFrame(report).transpose()
            st.dataframe(report_df)
            
            # Feature importance (for Random Forest)
            if model_type == "Random Forest":
                st.subheader("Feature Importance")
                importance_df = pd.DataFrame({
                    'Feature': feature_columns,
                    'Importance': model.feature_importances_
                }).sort_values('Importance', ascending=False)
                
                fig_importance = px.bar(importance_df, x='Importance', y='Feature', 
                                      orientation='h', title="Feature Importance")
                st.plotly_chart(fig_importance, use_container_width=True)
    
    # Page 3: Loan Prediction
    elif page == "Loan Prediction":
        st.header("🔮 Loan Prediction")
        
        # Check if model is trained
        if 'model' not in st.session_state:
            st.warning("Please train a model first in the 'Model Training' page.")
        else:
            st.subheader("Enter Applicant Information")
            
            # Create input form
            col1, col2 = st.columns(2)
            
            with col1:
                age = st.number_input("Age", min_value=18, max_value=80, value=30)
                annual_income = st.number_input("Annual Income ($)", min_value=20000, max_value=150000, value=60000)
                credit_score = st.number_input("Credit Score", min_value=300, max_value=850, value=650)
            
            with col2:
                employment_years = st.number_input("Employment Years", min_value=0, max_value=40, value=5)
                loan_amount = st.number_input("Loan Amount Requested ($)", min_value=5000, max_value=50000, value=15000)
            
            # Predict button
            if st.button("Predict Loan Approval", type="primary"):
                # Prepare input data
                input_data = np.array([[age, annual_income, credit_score, employment_years, loan_amount]])
                
                # Scale data if using logistic regression
                if st.session_state['model_type'] == "Logistic Regression":
                    input_data = st.session_state['scaler'].transform(input_data)
                
                # Make prediction
                prediction = st.session_state['model'].predict(input_data)[0]
                prediction_proba = st.session_state['model'].predict_proba(input_data)[0]
                
                # Display results
                st.markdown("---")
                st.subheader("Prediction Results")
                
                if prediction == 0:
                    st.success("🎉 **LOAN APPROVED**")
                    st.balloons()
                else:
                    st.error("❌ **LOAN REJECTED**")
                
                # Display probabilities
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Approval Probability", f"{prediction_proba[0]:.2%}")
                with col2:
                    st.metric("Rejection Probability", f"{prediction_proba[1]:.2%}")
                
                # Risk assessment
                st.subheader("Risk Assessment")
                if prediction_proba[1] < 0.3:
                    st.success("**Low Risk** - Strong candidate for loan approval")
                elif prediction_proba[1] < 0.6:
                    st.warning("**Medium Risk** - Additional review recommended")
                else:
                    st.error("**High Risk** - Loan application likely to be rejected")
    
    # Page 4: Model Performance
    elif page == "Model Performance":
        st.header("📈 Model Performance")
        
        if 'model' not in st.session_state:
            st.warning("Please train a model first in the 'Model Training' page.")
        else:
            st.subheader("Model Evaluation Metrics")
            
            # Re-evaluate model for detailed metrics
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            if st.session_state['model_type'] == "Logistic Regression":
                X_test_scaled = st.session_state['scaler'].transform(X_test)
                y_pred = st.session_state['model'].predict(X_test_scaled)
                y_pred_proba = st.session_state['model'].predict_proba(X_test_scaled)[:, 1]
            else:
                y_pred = st.session_state['model'].predict(X_test)
                y_pred_proba = st.session_state['model'].predict_proba(X_test)[:, 1]
            
            # Calculate metrics
            accuracy = accuracy_score(y_test, y_pred)
            
            # Display key metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Accuracy", f"{accuracy:.3f}")
            with col2:
                from sklearn.metrics import precision_score
                precision = precision_score(y_test, y_pred)
                st.metric("Precision", f"{precision:.3f}")
            with col3:
                from sklearn.metrics import recall_score
                recall = recall_score(y_test, y_pred)
                st.metric("Recall", f"{recall:.3f}")
            with col4:
                from sklearn.metrics import f1_score
                f1 = f1_score(y_test, y_pred)
                st.metric("F1 Score", f"{f1:.3f}")
            
            # ROC Curve
            st.subheader("ROC Curve")
            from sklearn.metrics import roc_curve, auc
            
            fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
            roc_auc = auc(fpr, tpr)
            
            fig_roc = go.Figure()
            fig_roc.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', name=f'ROC Curve (AUC = {roc_auc:.3f})'))
            fig_roc.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', name='Random Classifier', line=dict(dash='dash')))
            fig_roc.update_layout(
                title='ROC Curve',
                xaxis_title='False Positive Rate',
                yaxis_title='True Positive Rate'
            )
            st.plotly_chart(fig_roc, use_container_width=True)
            
            # Prediction distribution
            st.subheader("Prediction Distribution")
            fig_dist = go.Figure()
            fig_dist.add_trace(go.Histogram(x=y_pred_proba[y_test == 0], name='Approved (Actual)', meta=0.7))
            fig_dist.add_trace(go.Histogram(x=y_pred_proba[y_test == 1], name='Rejected (Actual)', meta=0.7))
            fig_dist.update_layout(
                title='Distribution of Predicted Probabilities',
                xaxis_title='Predicted Probability of Default',
                yaxis_title='Count',
                barmode='overlay'
            )
            st.plotly_chart(fig_dist, use_container_width=True)

else:
    st.error("Unable to load data. Please check if the CSV file exists and is properly formatted.")

# Footer
st.markdown("---")
st.markdown("*Built with Streamlit and Scikit-learn*")