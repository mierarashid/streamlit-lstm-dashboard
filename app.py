import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from tensorflow.keras.models import load_model
import joblib

# Load the saved model and scalers
@st.cache_resource
def load_saved_model():
    model = load_model('lstm_model.h5')
    X_scaler = joblib.load('X_scaler.pkl')
    y_scaler = joblib.load('y_scaler.pkl')
    return model, X_scaler, y_scaler

# Load model and scalers
model, X_scaler, y_scaler = load_saved_model()

# Page config
st.set_page_config(
    page_title="LSTM Model Dashboard",
    page_icon="📈",
    layout="wide"
)

# Sidebar
st.sidebar.title("Navigation")
page = st.sidebar.selectbox("Choose a page", ["Overview", "Make Predictions", "Model Performance", "Data Analysis"])

if page == "Overview":
    st.title("LSTM Model Dashboard")
    st.write("Welcome to the LSTM Model Dashboard. This dashboard allows you to:")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.info("📊 Visualize Historical Data")
    with col2:
        st.info("🔮 Make Real-time Predictions")
    with col3:
        st.info("📈 Monitor Model Performance")

elif page == "Make Predictions":
    st.title("Make Predictions")
    
    # Create input fields for your features
    st.subheader("Enter Feature Values")
    
    # Add input fields based on your features
    input_data = {}
    features = ['feature1', 'feature2', 'feature3']  # Replace with your actual features
    
    for feature in features:
        input_data[feature] = st.number_input(f"Enter {feature}", value=0.0)
    
    if st.button("Make Prediction"):
        # Prepare input data
        input_array = np.array(list(input_data.values())).reshape(1, -1)
        scaled_data = X_scaler.transform(input_array)
        lstm_input = scaled_data.reshape((1, 1, scaled_data.shape[1]))
        
        # Make prediction
        prediction = model.predict(lstm_input)
        final_prediction = y_scaler.inverse_transform(prediction)
        
        st.success(f"Predicted Value: {final_prediction[0][0]:.2f}")
        
        # Add confidence intervals or uncertainty visualization
        st.plotly_chart(go.Figure(go.Indicator(
            mode="gauge+number",
            value=final_prediction[0][0],
            gauge={'axis': {'range': [None, 100]}}
        )))

elif page == "Model Performance":
    st.title("Model Performance Metrics")
    
    # Add historical performance metrics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("MAE", "0.123")
    with col2:
        st.metric("MSE", "0.456")
    with col3:
        st.metric("R² Score", "0.789")
    
    # Add performance visualizations
    st.subheader("Actual vs Predicted Values")
    # Create sample data for visualization
    actual = np.random.rand(100)
    predicted = actual + np.random.normal(0, 0.1, 100)
    
    fig = px.scatter(x=actual, y=predicted, 
                    labels={'x': 'Actual Values', 'y': 'Predicted Values'},
                    trendline="ols")
    st.plotly_chart(fig)

elif page == "Data Analysis":
    st.title("Data Analysis")
    
    # Add file uploader for new data
    uploaded_file = st.file_uploader("Upload new data for analysis", type=['csv'])
    
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        
        # Show basic statistics
        st.subheader("Data Summary")
        st.write(df.describe())
        
        # Add interactive plots
        st.subheader("Feature Correlations")
        correlation_matrix = df.corr()
        fig = px.imshow(correlation_matrix, 
                       labels=dict(color="Correlation"),
                       color_continuous_scale="RdBu")
        st.plotly_chart(fig)
        
        # Time series plot if applicable
        if 'date' in df.columns:
            st.subheader("Time Series Analysis")
            fig = px.line(df, x='date', y=df.columns[1:])
            st.plotly_chart(fig)

# Add footer
st.markdown("---")
st.markdown("Created with ❤️ using Streamlit")
