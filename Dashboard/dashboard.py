import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import time

# Configure the page
st.set_page_config(
    page_title="Data Dashboard",
    page_icon="📊",
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
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #1f77b4;
    }
</style>
""", unsafe_allow_html=True)

# Main title
st.markdown('<h1 class="main-header">📊 Interactive Data Dashboard</h1>', unsafe_allow_html=True)

# Sidebar
st.sidebar.header("Configuration")
st.sidebar.markdown("---")

# Sidebar controls
data_source = st.sidebar.selectbox(
    "Choose Data Source",
    ["Sample Sales Data", "Random Dataset", "Upload CSV"]
)

if data_source == "Upload CSV":
    uploaded_file = st.sidebar.file_uploader("Choose a CSV file", type="csv")
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
    else:
        st.info("Please upload a CSV file to continue.")
        st.stop()
else:
    # Generate sample data
    if data_source == "Sample Sales Data":
        np.random.seed(42)
        dates = pd.date_range(start='2023-01-01', end='2024-12-31', freq='D')
        df = pd.DataFrame({
            'date': np.random.choice(dates, 1000),
            'product': np.random.choice(['Product A', 'Product B', 'Product C', 'Product D'], 1000),
            'region': np.random.choice(['North', 'South', 'East', 'West'], 1000),
            'sales': np.random.normal(1000, 300, 1000).round(2),
            'quantity': np.random.randint(1, 100, 1000)
        })
    else:
        np.random.seed(42)
        df = pd.DataFrame({
            'x': np.random.randn(500),
            'y': np.random.randn(500),
            'category': np.random.choice(['A', 'B', 'C'], 500),
            'value': np.random.uniform(0, 100, 500)
        })

# Sidebar filters
st.sidebar.markdown("### Filters")
if data_source == "Sample Sales Data":
    selected_products = st.sidebar.multiselect(
        "Select Products",
        options=df['product'].unique(),
        default=df['product'].unique()
    )
    selected_regions = st.sidebar.multiselect(
        "Select Regions",
        options=df['region'].unique(),
        default=df['region'].unique()
    )
    
    # Filter data
    df_filtered = df[
        (df['product'].isin(selected_products)) & 
        (df['region'].isin(selected_regions))
    ]
else:
    df_filtered = df

# Main content area
col1, col2, col3 = st.columns(3)

# Key metrics
if data_source == "Sample Sales Data":
    with col1:
        st.metric(
            label="Total Sales",
            value=f"${df_filtered['sales'].sum():,.2f}",
            delta=f"{df_filtered['sales'].sum() - df['sales'].mean() * len(df_filtered):,.2f}"
        )
    
    with col2:
        st.metric(
            label="Average Order Value",
            value=f"${df_filtered['sales'].mean():.2f}",
            delta=f"{df_filtered['sales'].mean() - df['sales'].mean():.2f}"
        )
    
    with col3:
        st.metric(
            label="Total Orders",
            value=f"{len(df_filtered):,}",
            delta=f"{len(df_filtered) - len(df)}"
        )

# Charts section
st.markdown("---")
st.subheader("📈 Data Visualizations")

# Create tabs for different chart types
tab1, tab2, tab3, tab4 = st.tabs(["📊 Bar Chart", "📈 Line Chart", "🔍 Scatter Plot", "📋 Data Table"])

with tab1:
    if data_source == "Sample Sales Data":
        # Bar chart
        sales_by_product = df_filtered.groupby('product')['sales'].sum().reset_index()
        fig_bar = px.bar(
            sales_by_product,
            x='product',
            y='sales',
            title='Sales by Product',
            color='sales',
            color_continuous_scale='viridis'
        )
        fig_bar.update_layout(showlegend=False)
        st.plotly_chart(fig_bar, use_container_width=True)
    else:
        # Bar chart for random data
        category_counts = df_filtered['category'].value_counts().reset_index()
        fig_bar = px.bar(
            category_counts,
            x='category',
            y='count',
            title='Count by Category'
        )
        st.plotly_chart(fig_bar, use_container_width=True)

with tab2:
    if data_source == "Sample Sales Data":
        # Line chart
        daily_sales = df_filtered.groupby('date')['sales'].sum().reset_index()
        daily_sales = daily_sales.sort_values('date')
        fig_line = px.line(
            daily_sales,
            x='date',
            y='sales',
            title='Daily Sales Trend'
        )
        st.plotly_chart(fig_line, use_container_width=True)
    else:
        # Line chart for random data
        df_sorted = df_filtered.sort_values('x')
        fig_line = px.line(
            df_sorted,
            x='x',
            y='value',
            title='Value Trend'
        )
        st.plotly_chart(fig_line, use_container_width=True)

with tab3:
    if data_source == "Sample Sales Data":
        # Scatter plot
        fig_scatter = px.scatter(
            df_filtered,
            x='quantity',
            y='sales',
            color='region',
            size='sales',
            title='Sales vs Quantity by Region',
            hover_data=['product']
        )
        st.plotly_chart(fig_scatter, use_container_width=True)
    else:
        # Scatter plot for random data
        fig_scatter = px.scatter(
            df_filtered,
            x='x',
            y='y',
            color='category',
            size='value',
            title='Scatter Plot by Category'
        )
        st.plotly_chart(fig_scatter, use_container_width=True)

with tab4:
    st.subheader("Raw Data")
    st.dataframe(df_filtered, use_container_width=True)
    
    # Download button
    csv = df_filtered.to_csv(index=False)
    st.download_button(
        label="Download CSV",
        data=csv,
        file_name=f"filtered_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv"
    )

# Interactive widgets section
st.markdown("---")
st.subheader("🎮 Interactive Elements")

col1, col2 = st.columns(2)

with col1:
    st.markdown("#### Input Widgets")
    
    # Text input
    user_name = st.text_input("Enter your name:", "User")
    
    # Number input
    number = st.number_input("Pick a number:", min_value=1, max_value=100, value=50)
    
    # Slider
    threshold = st.slider("Set threshold:", 0, 100, 25)
    
    # Date picker
    selected_date = st.date_input("Select a date:", datetime.now())
    
    # Checkbox
    show_advanced = st.checkbox("Show advanced options")
    
    if show_advanced:
        st.text_area("Advanced settings:", "Enter advanced configuration here...")

with col2:
    st.markdown("#### Results")
    
    st.write(f"Hello, {user_name}!")
    st.write(f"Your number: {number}")
    st.write(f"Threshold: {threshold}")
    st.write(f"Selected date: {selected_date}")
    
    # Progress bar
    if st.button("Run Process"):
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for i in range(100):
            progress_bar.progress(i + 1)
            status_text.text(f'Processing... {i+1}%')
            time.sleep(0.01)
        
        st.success("Process completed!")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; padding: 20px;'>
    <p>Built with Streamlit 🎈 | Data Dashboard Example</p>
</div>
""", unsafe_allow_html=True)

# Session state example
if 'counter' not in st.session_state:
    st.session_state.counter = 0

if st.button('Click me!'):
    st.session_state.counter += 1

st.write(f'Button clicked {st.session_state.counter} times')