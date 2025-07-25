import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import openai
import json
import io
import base64
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Auto Report Generator",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling
def load_css():
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    * {
        font-family: 'Inter', sans-serif;
    }
    
    /* Background with data science theme */
    .stApp {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        background-attachment: fixed;
    }
    
    .main-container {
        background: rgba(255, 255, 255, 0.95);
        backdrop-filter: blur(10px);
        border-radius: 20px;
        padding: 2rem;
        margin: 1rem;
        box-shadow: 0 15px 35px rgba(0, 0, 0, 0.1);
    }
    
    /* Header styling */
    .header-container {
        text-align: center;
        padding: 2rem 0;
        background: linear-gradient(135deg, #667eea, #764ba2);
        background-clip: text;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    /* Neon chat input styling */
    .chat-input {
        position: relative;
        margin: 1rem 0;
    }
    
    .stTextInput > div > div > input {
        background: rgba(255, 255, 255, 0.1);
        border: 2px solid #667eea;
        border-radius: 25px;
        padding: 15px 20px;
        color: #333;
        font-size: 16px;
        transition: all 0.3s ease;
        box-shadow: 0 0 20px rgba(102, 126, 234, 0.3);
    }
    
    .stTextInput > div > div > input:focus {
        border-color: #764ba2;
        box-shadow: 0 0 30px rgba(118, 75, 162, 0.5);
        transform: translateY(-2px);
    }
    
    /* KPI Card styling */
    .kpi-card {
        background: linear-gradient(135deg, #fff 0%, #f8f9ff 100%);
        border-radius: 15px;
        padding: 1.5rem;
        margin: 0.5rem;
        box-shadow: 0 8px 25px rgba(0, 0, 0, 0.08);
        border-left: 4px solid #667eea;
        transition: transform 0.3s ease;
    }
    
    .kpi-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 15px 35px rgba(0, 0, 0, 0.15);
    }
    
    .kpi-value {
        font-size: 2rem;
        font-weight: 700;
        color: #667eea;
        margin-bottom: 0.5rem;
    }
    
    .kpi-label {
        font-size: 0.9rem;
        color: #666;
        font-weight: 500;
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background: linear-gradient(180deg, #667eea 0%, #764ba2 100%);
    }
    
    .css-1d391kg .css-1v0mbdj {
        color: white;
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(135deg, #667eea, #764ba2);
        color: white;
        border: none;
        border-radius: 25px;
        padding: 0.75rem 2rem;
        font-weight: 600;
        transition: all 0.3s ease;
        box-shadow: 0 5px 15px rgba(102, 126, 234, 0.3);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 25px rgba(102, 126, 234, 0.4);
    }
    
    /* Chat message styling */
    .chat-message {
        padding: 1rem;
        border-radius: 15px;
        margin: 0.5rem 0;
        animation: fadeIn 0.5s ease-in;
    }
    
    .user-message {
        background: linear-gradient(135deg, #667eea, #764ba2);
        color: white;
        margin-left: 20%;
    }
    
    .bot-message {
        background: rgba(102, 126, 234, 0.1);
        border: 1px solid rgba(102, 126, 234, 0.2);
        margin-right: 20%;
    }
    
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    /* Theme toggle */
    .theme-toggle {
        position: fixed;
        top: 1rem;
        right: 1rem;
        z-index: 1000;
    }
    
    /* Dark theme overrides */
    .dark-theme {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
    }
    
    .dark-theme .main-container {
        background: rgba(20, 20, 40, 0.95);
        color: white;
    }
    
    .dark-theme .kpi-card {
        background: linear-gradient(135deg, #2a2a4a 0%, #1a1a3a 100%);
        color: white;
    }
    </style>
    """, unsafe_allow_html=True)

# Initialize session state
def init_session_state():
    if 'df' not in st.session_state:
        st.session_state.df = None
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    if 'dark_theme' not in st.session_state:
        st.session_state.dark_theme = False
    if 'openai_api_key' not in st.session_state:
        st.session_state.openai_api_key = ""

# OpenAI integration
class DataAnalysisBot:
    def __init__(self, api_key):
        openai.api_key = api_key
        
    def analyze_data(self, df, query):
        """Analyze data based on user query"""
        try:
            # Create data summary
            data_summary = self.create_data_summary(df)
            
            # Create prompt for OpenAI
            prompt = f"""
            You are a data analyst AI assistant. Based on the following dataset summary and user query, provide insights and suggest appropriate visualizations.
            
            Dataset Summary:
            {data_summary}
            
            User Query: {query}
            
            Please provide:
            1. Analysis of the query in context of the data
            2. Specific insights or findings
            3. Visualization recommendations (if applicable)
            4. Any statistical measures that might be relevant
            
            Keep the response concise and actionable.
            """
            
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=500,
                temperature=0.7
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            return f"Error analyzing data: {str(e)}"
    
    def create_data_summary(self, df):
        """Create a comprehensive data summary"""
        summary = f"""
        Dataset Shape: {df.shape[0]} rows, {df.shape[1]} columns
        
        Columns and Types:
        {df.dtypes.to_string()}
        
        Missing Values:
        {df.isnull().sum().to_string()}
        
        Numerical Columns Summary:
        {df.describe().to_string() if len(df.select_dtypes(include=[np.number]).columns) > 0 else 'No numerical columns'}
        
        Categorical Columns:
        {list(df.select_dtypes(include=['object']).columns)}
        """
        return summary

# Data loading functions
def load_data():
    """Load data from file upload or file path"""
    st.subheader("📂 Data Input")
    
    input_method = st.radio("Choose input method:", ["Upload CSV", "File Path"])
    
    if input_method == "Upload CSV":
        uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
                st.success(f"Dataset loaded successfully! Shape: {df.shape}")
                return df
            except Exception as e:
                st.error(f"Error loading file: {str(e)}")
                return None
    else:
        file_path = st.text_input("Enter CSV file path:")
        if file_path and st.button("Load Data"):
            try:
                df = pd.read_csv(file_path)
                st.success(f"Dataset loaded successfully! Shape: {df.shape}")
                return df
            except Exception as e:
                st.error(f"Error loading file: {str(e)}")
                return None
    
    return None

# EDA functions
def display_kpi_cards(df):
    """Display KPI cards with key statistics"""
    st.subheader("📊 Key Performance Indicators")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-value">{df.shape[0]:,}</div>
            <div class="kpi-label">Total Rows</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-value">{df.shape[1]}</div>
            <div class="kpi-label">Total Columns</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        missing_pct = (df.isnull().sum().sum() / (df.shape[0] * df.shape[1])) * 100
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-value">{missing_pct:.1f}%</div>
            <div class="kpi-label">Missing Values</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        numeric_cols = len(df.select_dtypes(include=[np.number]).columns)
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-value">{numeric_cols}</div>
            <div class="kpi-label">Numeric Columns</div>
        </div>
        """, unsafe_allow_html=True)

def perform_eda(df):
    """Perform comprehensive EDA"""
    st.subheader("🔍 Exploratory Data Analysis")
    
    # Data overview
    with st.expander("Data Overview", expanded=True):
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**First 5 rows:**")
            st.dataframe(df.head())
        
        with col2:
            st.write("**Data Types:**")
            st.dataframe(df.dtypes.to_frame(name='Data Type'))
    
    # Statistical summary
    with st.expander("Statistical Summary"):
        numeric_df = df.select_dtypes(include=[np.number])
        if not numeric_df.empty:
            st.dataframe(numeric_df.describe())
        else:
            st.write("No numeric columns found for statistical summary.")
    
    # Missing values analysis
    with st.expander("Missing Values Analysis"):
        missing_data = df.isnull().sum()
        missing_data = missing_data[missing_data > 0].sort_values(ascending=False)
        
        if not missing_data.empty:
            fig = px.bar(x=missing_data.index, y=missing_data.values,
                        title="Missing Values by Column",
                        labels={'x': 'Columns', 'y': 'Missing Count'})
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.success("No missing values found in the dataset!")
    
    # Correlation analysis
    with st.expander("Correlation Analysis"):
        numeric_df = df.select_dtypes(include=[np.number])
        if len(numeric_df.columns) > 1:
            corr_matrix = numeric_df.corr()
            
            fig = px.imshow(corr_matrix, 
                          text_auto=True, 
                          aspect="auto",
                          title="Correlation Heatmap",
                          color_continuous_scale="RdBu")
            st.plotly_chart(fig, use_container_width=True)
            
            # Top correlations
            corr_pairs = []
            for i in range(len(corr_matrix.columns)):
                for j in range(i+1, len(corr_matrix.columns)):
                    corr_pairs.append((
                        corr_matrix.columns[i],
                        corr_matrix.columns[j],
                        corr_matrix.iloc[i, j]
                    ))
            
            corr_pairs.sort(key=lambda x: abs(x[2]), reverse=True)
            
            st.write("**Top 5 Correlations:**")
            for i, (col1, col2, corr) in enumerate(corr_pairs[:5]):
                st.write(f"{i+1}. {col1} ↔ {col2}: {corr:.3f}")
        else:
            st.write("Need at least 2 numeric columns for correlation analysis.")

def create_visualizations(df, query):
    """Create visualizations based on query"""
    st.subheader("📈 Visualizations")
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    
    viz_type = st.selectbox("Choose visualization type:", 
                           ["Distribution Plot", "Scatter Plot", "Box Plot", "Bar Chart", "Pair Plot"])
    
    if viz_type == "Distribution Plot" and numeric_cols:
        col = st.selectbox("Select column:", numeric_cols)
        fig = px.histogram(df, x=col, title=f"Distribution of {col}")
        st.plotly_chart(fig, use_container_width=True)
    
    elif viz_type == "Scatter Plot" and len(numeric_cols) >= 2:
        col1, col2 = st.columns(2)
        with col1:
            x_col = st.selectbox("X-axis:", numeric_cols)
        with col2:
            y_col = st.selectbox("Y-axis:", [col for col in numeric_cols if col != x_col])
        
        color_col = st.selectbox("Color by (optional):", ["None"] + categorical_cols)
        color_col = None if color_col == "None" else color_col
        
        fig = px.scatter(df, x=x_col, y=y_col, color=color_col, 
                        title=f"{y_col} vs {x_col}")
        st.plotly_chart(fig, use_container_width=True)
    
    elif viz_type == "Box Plot" and numeric_cols:
        y_col = st.selectbox("Select numeric column:", numeric_cols)
        x_col = st.selectbox("Group by (optional):", ["None"] + categorical_cols)
        x_col = None if x_col == "None" else x_col
        
        fig = px.box(df, x=x_col, y=y_col, title=f"Box Plot of {y_col}")
        st.plotly_chart(fig, use_container_width=True)
    
    elif viz_type == "Bar Chart" and categorical_cols:
        col = st.selectbox("Select categorical column:", categorical_cols)
        value_counts = df[col].value_counts().head(10)
        
        fig = px.bar(x=value_counts.index, y=value_counts.values,
                    title=f"Top 10 values in {col}",
                    labels={'x': col, 'y': 'Count'})
        st.plotly_chart(fig, use_container_width=True)
    
    elif viz_type == "Pair Plot" and numeric_cols:
        if len(numeric_cols) > 5:
            selected_cols = st.multiselect("Select columns (max 5):", numeric_cols, default=numeric_cols[:4])
        else:
            selected_cols = numeric_cols
        
        if selected_cols:
            fig = px.scatter_matrix(df[selected_cols], title="Pair Plot")
            st.plotly_chart(fig, use_container_width=True)

# Chat interface
def chat_interface(df):
    """AI Chat interface"""
    st.subheader("🤖 AI Data Analysis Chat")
    
    # API Key input
    api_key = st.text_input("Enter your OpenAI API Key:", type="password", 
                           value=st.session_state.openai_api_key)
    
    if api_key:
        st.session_state.openai_api_key = api_key
        bot = DataAnalysisBot(api_key)
        
        # Chat input
        st.markdown('<div class="chat-input">', unsafe_allow_html=True)
        user_query = st.text_input("Ask me anything about your data:", 
                                 placeholder="e.g., 'Show me the correlation between age and income'")
        st.markdown('</div>', unsafe_allow_html=True)
        
        if user_query and st.button("Send"):
            # Add user message to chat history
            st.session_state.chat_history.append({"role": "user", "message": user_query})
            
            # Get AI response
            with st.spinner("Analyzing..."):
                response = bot.analyze_data(df, user_query)
                st.session_state.chat_history.append({"role": "bot", "message": response})
        
        # Display chat history
        for chat in st.session_state.chat_history:
            if chat["role"] == "user":
                st.markdown(f"""
                <div class="chat-message user-message">
                    <strong>You:</strong> {chat["message"]}
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="chat-message bot-message">
                    <strong>AI Assistant:</strong> {chat["message"]}
                </div>
                """, unsafe_allow_html=True)
        
        # Suggest common queries
        st.write("**Quick Actions:**")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("Show data summary"):
                summary_query = "Give me a comprehensive summary of this dataset"
                st.session_state.chat_history.append({"role": "user", "message": summary_query})
                response = bot.analyze_data(df, summary_query)
                st.session_state.chat_history.append({"role": "bot", "message": response})
                st.experimental_rerun()
        
        with col2:
            if st.button("Find correlations"):
                corr_query = "What are the strongest correlations in this dataset?"
                st.session_state.chat_history.append({"role": "user", "message": corr_query})
                response = bot.analyze_data(df, corr_query)
                st.session_state.chat_history.append({"role": "bot", "message": response})
                st.experimental_rerun()
        
        with col3:
            if st.button("Identify outliers"):
                outlier_query = "Help me identify potential outliers in the numeric columns"
                st.session_state.chat_history.append({"role": "user", "message": outlier_query})
                response = bot.analyze_data(df, outlier_query)
                st.session_state.chat_history.append({"role": "bot", "message": response})
                st.experimental_rerun()

# Export functionality
def export_report(df):
    """Export EDA summary as downloadable report"""
    st.subheader("📄 Export Report")
    
    if st.button("Generate Report"):
        # Create report content
        report_content = f"""
# Auto-Generated EDA Report
Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Dataset Overview
- **Shape:** {df.shape[0]} rows, {df.shape[1]} columns
- **Missing Values:** {df.isnull().sum().sum()} total

## Column Information
{df.dtypes.to_string()}

## Statistical Summary
{df.describe().to_string() if len(df.select_dtypes(include=[np.number]).columns) > 0 else 'No numerical columns for summary'}

## Missing Values by Column
{df.isnull().sum().to_string()}

## Data Quality Issues
- Duplicate rows: {df.duplicated().sum()}
- Completely null rows: {df.isnull().all(axis=1).sum()}

---
*Report generated by Auto Report Generator*
        """
        
        # Convert to downloadable format
        st.download_button(
            label="Download Markdown Report",
            data=report_content,
            file_name=f"EDA_Report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
            mime="text/markdown"
        )

# Sidebar content
def setup_sidebar(df):
    """Setup sidebar with dataset information"""
    st.sidebar.title("📊 Auto Report Generator")
    
    # Theme toggle
    if st.sidebar.button("🌓 Toggle Theme"):
        st.session_state.dark_theme = not st.session_state.dark_theme
    
    if df is not None:
        st.sidebar.subheader("Dataset Info")
        st.sidebar.write(f"**Rows:** {df.shape[0]:,}")
        st.sidebar.write(f"**Columns:** {df.shape[1]}")
        st.sidebar.write(f"**Memory Usage:** {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
        
        # Column information
        with st.sidebar.expander("Column Details"):
            for col in df.columns:
                dtype = str(df[col].dtype)
                null_count = df[col].isnull().sum()
                st.write(f"**{col}**")
                st.write(f"Type: {dtype}")
                st.write(f"Nulls: {null_count}")
                st.write("---")
    
    # About section
    with st.sidebar.expander("About This App"):
        st.write("""
        **Auto Report Generator** is an AI-powered data analysis tool that helps you:
        
        - 📊 Perform automated EDA
        - 🤖 Chat with your data using AI
        - 📈 Generate interactive visualizations
        - 📄 Export comprehensive reports
        
        Built with Streamlit, OpenAI, and modern data science libraries.
        """)

# Main application
def main():
    # Load CSS and initialize session state
    load_css()
    init_session_state()
    
    # Header
    st.markdown("""
    <div class="header-container">
        <h1>📊 Auto Report Generator</h1>
        <p>AI-Powered Exploratory Data Analysis & Reporting Tool</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Main container
    st.markdown('<div class="main-container">', unsafe_allow_html=True)
    
    # Load data
    df = load_data()
    
    if df is not None:
        st.session_state.df = df
        
        # Setup sidebar
        setup_sidebar(df)
        
        # Display KPI cards
        display_kpi_cards(df)
        
        # Create tabs for different sections
        tab1, tab2, tab3, tab4 = st.tabs(["📊 EDA", "📈 Visualizations", "🤖 AI Chat", "📄 Export"])
        
        with tab1:
            perform_eda(df)
        
        with tab2:
            create_visualizations(df, "")
        
        with tab3:
            chat_interface(df)
        
        with tab4:
            export_report(df)
    
    else:
        # Setup sidebar without data
        setup_sidebar(None)
        
        st.info("👆 Please upload a CSV file or provide a file path to get started!")
        
        # Demo section
        st.subheader("🚀 Features")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
            **🔍 Automated EDA**
            - Data profiling
            - Statistical summaries
            - Missing value analysis
            - Correlation matrices
            """)
        
        with col2:
            st.markdown("""
            **🤖 AI Assistant**
            - Natural language queries
            - Intelligent insights
            - Data interpretation
            - Recommendation engine
            """)
        
        with col3:
            st.markdown("""
            **📊 Rich Visualizations**
            - Interactive charts
            - Multiple plot types
            - Customizable views
            - Export capabilities
            """)
    
    st.markdown('</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()