import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Set page configuration
st.set_page_config(
    page_title="Credit Card Fraud Detection",
    page_icon="🔐",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
    }
    .fraud-alert {
        background-color: #ffebee;
        color: #c62828;
        padding: 1rem;
        border-radius: 5px;
        border-left: 4px solid #c62828;
    }
    .safe-alert {
        background-color: #e8f5e8;
        color: #2e7d32;
        padding: 1rem;
        border-radius: 5px;
        border-left: 4px solid #2e7d32;
    }
</style>
""", unsafe_allow_html=True)

# Generate sample data
@st.cache_data
def generate_sample_data(n_samples=5000):
    """Generate realistic credit card transaction data"""
    np.random.seed(42)
    
    # Generate normal transactions
    normal_transactions = int(n_samples * 0.98)  # 98% normal
    fraud_transactions = n_samples - normal_transactions  # 2% fraud
    
    # Normal transaction features
    normal_data = {
        'amount': np.random.lognormal(3, 1, normal_transactions),
        'time_since_last_transaction': np.random.exponential(2, normal_transactions),
        'merchant_category': np.random.choice(['grocery', 'gas', 'restaurant', 'retail', 'online'], normal_transactions),
        'location_risk_score': np.random.beta(2, 8, normal_transactions),
        'time_of_day': np.random.normal(12, 6, normal_transactions) % 24,
        'day_of_week': np.random.choice(range(7), normal_transactions),
        'is_weekend': np.random.choice([0, 1], normal_transactions, p=[0.7, 0.3]),
        'customer_age': np.random.normal(45, 15, normal_transactions),
        'account_age_days': np.random.normal(1000, 500, normal_transactions),
        'previous_failed_attempts': np.random.poisson(0.1, normal_transactions),
        'is_fraud': np.zeros(normal_transactions)
    }
    
    # Fraudulent transaction features (different patterns)
    fraud_data = {
        'amount': np.concatenate([
            np.random.lognormal(2, 0.5, fraud_transactions//2),  # Small amounts
            np.random.lognormal(6, 1, fraud_transactions//2)     # Large amounts
        ]),
        'time_since_last_transaction': np.random.exponential(0.5, fraud_transactions),
        'merchant_category': np.random.choice(['online', 'retail', 'gas'], fraud_transactions),
        'location_risk_score': np.random.beta(8, 2, fraud_transactions),
        'time_of_day': np.random.choice([2, 3, 4, 22, 23], fraud_transactions),
        'day_of_week': np.random.choice(range(7), fraud_transactions),
        'is_weekend': np.random.choice([0, 1], fraud_transactions, p=[0.4, 0.6]),
        'customer_age': np.random.normal(35, 20, fraud_transactions),
        'account_age_days': np.random.normal(200, 300, fraud_transactions),
        'previous_failed_attempts': np.random.poisson(2, fraud_transactions),
        'is_fraud': np.ones(fraud_transactions)
    }
    
    # Combine data
    data = {}
    for key in normal_data.keys():
        data[key] = np.concatenate([normal_data[key], fraud_data[key]])
    
    df = pd.DataFrame(data)
    
    # Add some derived features
    df['amount_zscore'] = (df['amount'] - df['amount'].mean()) / df['amount'].std()
    df['velocity_risk'] = 1 / (df['time_since_last_transaction'] + 0.1)
    df['late_night_transaction'] = ((df['time_of_day'] >= 22) | (df['time_of_day'] <= 6)).astype(int)
    
    return df

# Load and prepare data
@st.cache_data
def load_data():
    df = generate_sample_data()
    return df

# Train model
@st.cache_resource
def train_model(df):
    """Train fraud detection model"""
    feature_columns = ['amount', 'time_since_last_transaction', 'location_risk_score', 
                      'time_of_day', 'day_of_week', 'is_weekend', 'customer_age', 
                      'account_age_days', 'previous_failed_attempts', 'amount_zscore', 
                      'velocity_risk', 'late_night_transaction']
    
    X = df[feature_columns]
    y = df['is_fraud']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train model
    model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
    model.fit(X_train_scaled, y_train)
    
    # Get predictions and probabilities
    y_pred = model.predict(X_test_scaled)
    y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
    
    return model, scaler, X_test, y_test, y_pred, y_pred_proba, feature_columns

# Main app
def main():
    st.markdown('<h1 class="main-header">🔐 Credit Card Fraud Detection System</h1>', unsafe_allow_html=True)
    
    # Load data and train model
    df = load_data()
    model, scaler, X_test, y_test, y_pred, y_pred_proba, feature_columns = train_model(df)
    
    # Sidebar
    st.sidebar.header("Navigation")
    page = st.sidebar.selectbox("Choose a page", ["Dashboard", "Single Transaction Check", "Batch Analysis", "Model Performance"])
    
    if page == "Dashboard":
        dashboard_page(df, model, scaler, feature_columns)
    elif page == "Single Transaction Check":
        single_transaction_page(model, scaler, feature_columns)
    elif page == "Batch Analysis":
        batch_analysis_page(df, model, scaler, feature_columns)
    elif page == "Model Performance":
        model_performance_page(X_test, y_test, y_pred, y_pred_proba, model, feature_columns)

def dashboard_page(df, model, scaler, feature_columns):
    """Main dashboard with overview statistics"""
    st.header("📊 Fraud Detection Dashboard")
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Transactions", f"{len(df):,}")
    
    with col2:
        fraud_count = df['is_fraud'].sum()
        fraud_rate = (fraud_count / len(df)) * 100
        st.metric("Fraud Transactions", f"{fraud_count:,}", f"{fraud_rate:.2f}%")
    
    with col3:
        avg_amount = df['amount'].mean()
        st.metric("Avg Transaction Amount", f"${avg_amount:.2f}")
    
    with col4:
        fraud_amount = df[df['is_fraud'] == 1]['amount'].sum()
        st.metric("Total Fraud Amount", f"${fraud_amount:,.2f}")
    
    # Charts
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Transaction Amount Distribution")
        fig = px.histogram(df, x='amount', color='is_fraud', 
                          title='Transaction Amount Distribution by Fraud Status',
                          labels={'is_fraud': 'Fraud Status'})
        fig.update_layout(xaxis_title="Amount ($)", yaxis_title="Count")
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Fraud by Time of Day")
        hourly_fraud = df.groupby('time_of_day')['is_fraud'].agg(['sum', 'count', 'mean']).reset_index()
        fig = px.bar(hourly_fraud, x='time_of_day', y='mean',
                     title='Fraud Rate by Hour of Day',
                     labels={'mean': 'Fraud Rate', 'time_of_day': 'Hour'})
        st.plotly_chart(fig, use_container_width=True)
    
    # Risk factors
    st.subheader("🎯 Key Risk Factors")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Merchant Category Risk")
        merchant_risk = df.groupby('merchant_category')['is_fraud'].agg(['sum', 'count', 'mean']).reset_index()
        merchant_risk.columns = ['merchant_category', 'fraud_count', 'total_transactions', 'fraud_rate']
        st.dataframe(merchant_risk.sort_values('fraud_rate', ascending=False))
    
    with col2:
        st.subheader("Location Risk Distribution")
        fig = px.box(df, x='is_fraud', y='location_risk_score',
                     title='Location Risk Score by Fraud Status')
        st.plotly_chart(fig, use_container_width=True)

def single_transaction_page(model, scaler, feature_columns):
    """Page for checking individual transactions"""
    st.header("🔍 Single Transaction Fraud Check")
    
    st.write("Enter transaction details to check for fraud probability:")
    
    col1, col2 = st.columns(2)
    
    with col1:
        amount = st.number_input("Transaction Amount ($)", min_value=0.01, value=100.0, step=0.01)
        time_since_last = st.number_input("Hours Since Last Transaction", min_value=0.0, value=2.0, step=0.1)
        location_risk = st.slider("Location Risk Score", 0.0, 1.0, 0.2, 0.01)
        time_of_day = st.slider("Time of Day (24h)", 0, 23, 14)
        day_of_week = st.selectbox("Day of Week", [0, 1, 2, 3, 4, 5, 6], format_func=lambda x: ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'][x])
        is_weekend = st.checkbox("Is Weekend")
    
    with col2:
        customer_age = st.number_input("Customer Age", min_value=18, max_value=100, value=35)
        account_age_days = st.number_input("Account Age (Days)", min_value=0, value=365)
        previous_failed = st.number_input("Previous Failed Attempts", min_value=0, value=0)
        merchant_category = st.selectbox("Merchant Category", ['grocery', 'gas', 'restaurant', 'retail', 'online'])
    
    if st.button("Check Fraud Risk", type="primary"):
        # Prepare input data
        input_data = pd.DataFrame({
            'amount': [amount],
            'time_since_last_transaction': [time_since_last],
            'location_risk_score': [location_risk],
            'time_of_day': [time_of_day],
            'day_of_week': [day_of_week],
            'is_weekend': [1 if is_weekend else 0],
            'customer_age': [customer_age],
            'account_age_days': [account_age_days],
            'previous_failed_attempts': [previous_failed],
            'amount_zscore': [(amount - 150) / 200],  # Approximate normalization
            'velocity_risk': [1 / (time_since_last + 0.1)],
            'late_night_transaction': [1 if (time_of_day >= 22 or time_of_day <= 6) else 0]
        })
        
        # Make prediction
        input_scaled = scaler.transform(input_data[feature_columns])
        fraud_probability = model.predict_proba(input_scaled)[0][1]
        is_fraud_pred = model.predict(input_scaled)[0]
        
        # Display results
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Fraud Risk Assessment")
            if fraud_probability > 0.5:
                st.markdown(f'<div class="fraud-alert">⚠️ HIGH RISK: {fraud_probability:.1%} fraud probability</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="safe-alert">✅ LOW RISK: {fraud_probability:.1%} fraud probability</div>', unsafe_allow_html=True)
        
        with col2:
            st.subheader("Risk Factors")
            risk_factors = []
            if location_risk > 0.7:
                risk_factors.append("High-risk location")
            if time_of_day >= 22 or time_of_day <= 6:
                risk_factors.append("Late night transaction")
            if time_since_last < 0.5:
                risk_factors.append("High transaction velocity")
            if previous_failed > 0:
                risk_factors.append("Previous failed attempts")
            if amount > 1000:
                risk_factors.append("Large transaction amount")
            
            if risk_factors:
                for factor in risk_factors:
                    st.write(f"• {factor}")
            else:
                st.write("No significant risk factors detected")

def batch_analysis_page(df, model, scaler, feature_columns):
    """Page for analyzing batches of transactions"""
    st.header("📋 Batch Transaction Analysis")
    
    st.write("Analyze recent transactions for fraud patterns:")
    
    # Sample recent transactions
    sample_size = st.slider("Number of transactions to analyze", 50, 500, 100)
    sample_df = df.sample(n=sample_size, random_state=42)
    
    # Make predictions for the sample
    X_sample = sample_df[feature_columns]
    X_sample_scaled = scaler.transform(X_sample)
    fraud_probabilities = model.predict_proba(X_sample_scaled)[:, 1]
    fraud_predictions = model.predict(X_sample_scaled)
    
    # Add predictions to dataframe
    sample_df = sample_df.copy()
    sample_df['fraud_probability'] = fraud_probabilities
    sample_df['predicted_fraud'] = fraud_predictions
    
    # Summary statistics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        actual_fraud = sample_df['is_fraud'].sum()
        st.metric("Actual Fraud Cases", actual_fraud)
    
    with col2:
        predicted_fraud = sample_df['predicted_fraud'].sum()
        st.metric("Predicted Fraud Cases", predicted_fraud)
    
    with col3:
        high_risk = (sample_df['fraud_probability'] > 0.3).sum()
        st.metric("High Risk Transactions", high_risk)
    
    # Show high-risk transactions
    st.subheader("🚨 High-Risk Transactions")
    high_risk_df = sample_df[sample_df['fraud_probability'] > 0.3].sort_values('fraud_probability', ascending=False)
    
    if not high_risk_df.empty:
        display_cols = ['amount', 'merchant_category', 'location_risk_score', 'time_of_day', 
                       'fraud_probability', 'is_fraud', 'predicted_fraud']
        st.dataframe(high_risk_df[display_cols].head(20))
    else:
        st.write("No high-risk transactions detected in this sample.")
    
    # Visualizations
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Fraud Probability Distribution")
        fig = px.histogram(sample_df, x='fraud_probability', 
                          title='Distribution of Fraud Probabilities')
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Risk vs Amount")
        fig = px.scatter(sample_df, x='amount', y='fraud_probability',
                        color='is_fraud', title='Fraud Risk vs Transaction Amount')
        st.plotly_chart(fig, use_container_width=True)

def model_performance_page(X_test, y_test, y_pred, y_pred_proba, model, feature_columns):
    """Page showing model performance metrics"""
    st.header("📈 Model Performance Analysis")
    
    # Performance metrics
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        accuracy = accuracy_score(y_test, y_pred)
        st.metric("Accuracy", f"{accuracy:.3f}")
    
    with col2:
        precision = precision_score(y_test, y_pred)
        st.metric("Precision", f"{precision:.3f}")
    
    with col3:
        recall = recall_score(y_test, y_pred)
        st.metric("Recall", f"{recall:.3f}")
    
    with col4:
        f1 = f1_score(y_test, y_pred)
        st.metric("F1 Score", f"{f1:.3f}")
    
    # ROC Curve
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("ROC Curve")
        from sklearn.metrics import roc_curve
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        auc_score = roc_auc_score(y_test, y_pred_proba)
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=fpr, y=tpr, name=f'ROC Curve (AUC = {auc_score:.3f})'))
        fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', name='Random'))
        fig.update_layout(title='ROC Curve', xaxis_title='False Positive Rate', yaxis_title='True Positive Rate')
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Confusion Matrix")
        cm = confusion_matrix(y_test, y_pred)
        fig = px.imshow(cm, text_auto=True, aspect="auto",
                       labels=dict(x="Predicted", y="Actual"),
                       title="Confusion Matrix")
        st.plotly_chart(fig, use_container_width=True)
    
    # Feature importance
    st.subheader("Feature Importance")
    importance_df = pd.DataFrame({
        'feature': feature_columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    fig = px.bar(importance_df, x='importance', y='feature', orientation='h',
                 title='Feature Importance in Fraud Detection')
    st.plotly_chart(fig, use_container_width=True)
    
    # Classification report
    st.subheader("Detailed Classification Report")
    report = classification_report(y_test, y_pred, output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    st.dataframe(report_df)

if __name__ == "__main__":
    main()