import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from tensorflow.keras.models import load_model
import joblib

# Page config
st.set_page_config(
    page_title="LSTM Model Dashboard",
    page_icon="📈",
    layout="wide"
)

# Define your features and target
features = ['total_ghg_savings', 'total_charging_sec', '7_rolling_avg', '30_rolling_avg', 'lag_1', 'lag_2', 'lag_3', 'lag_7', 'day_of_week', 'is_weekend', 'month'] 
target_variable = 'total_energy'

# Load the saved model and scalers
@st.cache_resource
def load_saved_model():
    model = load_model('lstm_model.h5')
    X_scaler = joblib.load('X_scaler.pkl')
    y_scaler = joblib.load('y_scaler.pkl')
    return model, X_scaler, y_scaler

# Load model and scalers
model, X_scaler, y_scaler = load_saved_model()

# Sidebar
st.sidebar.title("Navigation")
page = st.sidebar.selectbox("Choose a page", ["Overview", "Make Predictions", "Model Performance", "Data Analysis"])

if page == "Overview":
    st.title("EV Energy Demand Prediction Dashboard")
    st.write("""
    This dashboard provides tools to predict the energy demand of EVs based on historical data. 
    The model takes into account various features including:
    - GHG savings
    - Charging duration
    - Rolling averages
    - Time-based patterns
    """)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.info("📊 Upload and analyze your historical charging data")
    with col2:
        st.info("🔮 Get predictions for future energy demand")
    with col3:
        st.info("📈 Evaluate model performance and accuracy")

elif page == "Make Predictions":
    st.title("Make Predictions")
    
    # Add file uploader
    st.subheader("Upload Your Data")
    uploaded_file = st.file_uploader("Upload a CSV file", type=['csv'])
    
    if uploaded_file is not None:
        try:
            # Read the CSV file with date parsing
            input_df = pd.read_csv(uploaded_file)
            
            # Convert date column to datetime if it exists
            if date_column in input_df.columns:
                input_df[date_column] = pd.to_datetime(input_df[date_column])
                
            # Show the uploaded data
            st.subheader("Uploaded Data Preview")
            st.dataframe(input_df.head())
            
            if st.button("Make Predictions"):
                # Extract features for prediction (excluding date column)
                features_df = input_df[features]
                
                # Scale the input data
                input_scaled = X_scaler.transform(features_df)
                
                # Reshape for LSTM
                input_lstm = input_scaled.reshape((input_scaled.shape[0], 1, input_scaled.shape[1]))
                
                # Make predictions
                predictions = model.predict(input_lstm)
                final_predictions = y_scaler.inverse_transform(predictions)
                
                # Create results dataframe with date
                results_df = pd.DataFrame({
                    'Date': input_df[date_column],
                    'Actual': input_df[target_variable].values,
                    'Predictions': final_predictions.flatten()
                })
                
                # Display results
                st.subheader("Predictions")
                st.dataframe(results_df)
                
                # Download button for predictions
                csv = results_df.to_csv(index=False)
                st.download_button(
                    label="Download Predictions",
                    data=csv,
                    file_name="predictions.csv",
                    mime="text/csv"
                )
                
                # Visualizations with dates
                st.subheader("Actual vs Predicted Values Over Time")
                # Time series plot
                fig_time = px.line(results_df, 
                                 x='Date',
                                 y=['Actual', 'Predictions'],
                                 title='Actual vs Predicted Values Over Time',
                                 labels={'value': 'Energy Consumption', 
                                        'variable': 'Type',
                                        'Date': 'Date'},
                                 template='plotly')
                fig_time.update_layout(
                    xaxis_title="Date",
                    yaxis_title="Energy Consumption",
                    legend_title="Type",
                    hovermode='x unified'
                )
                st.plotly_chart(fig_time)

                # Scatter plot
                fig_scatter = px.scatter(results_df, 
                                       x='Actual', 
                                       y='Predictions',
                                       title='Prediction Correlation',
                                       labels={'Actual': 'Actual Values', 
                                              'Predictions': 'Predicted Values'})
                
                # Add perfect prediction line
                fig_scatter.add_trace(
                    go.Scatter(x=[results_df['Actual'].min(), results_df['Actual'].max()],
                              y=[results_df['Actual'].min(), results_df['Actual'].max()],
                              mode='lines',
                              name='Perfect Prediction',
                              line=dict(dash='dash', color='red'))
                )
                st.plotly_chart(fig_scatter)

        except Exception as e:
            st.error(f"Error processing the file: {str(e)}")
            st.write("Please make sure your CSV file has the following columns:")
            st.write(f"Date column: {date_column}")
            st.write("Feature columns:", ", ".join(features))
            st.write(f"Target column: {target_variable}")
    
    # Update template section
    st.markdown("---")
    st.subheader("CSV Template")
    st.write("Your CSV should have the following columns:")
    
    # Create sample template with date
    sample_df = pd.DataFrame(columns=[date_column] + features + [target_variable])
    st.dataframe(sample_df)
    
    # Sample data description
    st.write("""
    ### Data Format Requirements:
    - **Date**: Should be in YYYY-MM-DD format
    - **Features**: All numeric values
    - **Target**: The actual energy consumption values
    """)
    
    # Add download template button
    csv_template = convert_df_to_csv(sample_df)
    st.download_button(
        label="Download CSV Template",
        data=csv_template,
        file_name="template.csv",
        mime="text/csv"
    )
    # Add example template
    st.markdown("---")
    st.subheader("CSV Template")
    st.write("Your CSV should have the following columns:")
    
    # Create a sample dataframe with your features
    sample_df = pd.DataFrame(columns=['feature1', 'feature2', 'feature3'])  # Replace with your actual features
    st.dataframe(sample_df)
    
    # Add download template button
    @st.cache_data
    def convert_df_to_csv(df):
        return df.to_csv(index=False)
    
    csv_template = convert_df_to_csv(sample_df)
    st.download_button(
        label="Download CSV Template",
        data=csv_template,
        file_name="template.csv",
        mime="text/csv"
    )

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
