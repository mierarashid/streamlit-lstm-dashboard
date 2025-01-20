import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from tensorflow.keras.models import load_model
import joblib

# Page config
st.set_page_config(
    page_title="EV Energy Dashboard",
    page_icon="📈",
    layout="wide"
)

st.markdown("""
    <style>
    /* Main background and text colors */
    .stApp {
        background: linear-gradient(to bottom right, #f8f9fa, #e9ecef);
    }
    
    .main-title {
        color: #1E293B;
        font-size: 2.5rem;
        font-weight: 700;
        margin-bottom: 1rem;
    }
    
    .sub-text {
        color: #64748B;
        font-size: 1.1rem;
        line-height: 1.5;
    }
    
    /* Upload section styling */
    .upload-section {
        background: white;
        padding: 2rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin-bottom: 2rem;
    }
    
    /* Data preview section */
    .stDataFrame {
        background: white;
        padding: 1rem;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
        background-color: white;
        padding: 0.5rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    .stTabs [data-baseweb="tab"] {
        color: #64748B;
        font-weight: 500;
    }
    
    .stTabs [aria-selected="true"] {
        color: #2563EB;
        border-bottom-color: #2563EB;
    }
    
    /* Button styling */
    .stButton > button {
        background-color: #2563EB;
        color: white;
        border-radius: 6px;
        padding: 0.5rem 1rem;
        font-weight: 500;
        border: none;
        box-shadow: 0 2px 4px rgba(37,99,235,0.2);
        transition: all 0.2s ease;
    }
    
    /* File uploader styling */
    [data-testid="stFileUploader"] {
        background-color: white;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    </style>
""", unsafe_allow_html=True)

# Wrap the title and description in styled divs
st.markdown('<div class="main-header">', unsafe_allow_html=True)
st.markdown('<h1 class="main-title">EV Energy Demand Analysis & Prediction Dashboard</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-text">This dashboard provides tools to analyze your EV charging data and predict future energy demand. Upload your data below to get started.</p>', unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)


# Define your features and target
date_column = 'date' 
features = ['total_ghg_savings', 'total_charging_sec', '7_rolling_avg', '30_rolling_avg', 
            'lag_1', 'lag_2', 'lag_3', 'lag_7', 'day_of_week', 'is_weekend', 'month'] 
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

# Main title
#st.title("EV Energy Demand Analysis & Prediction Dashboard")
#st.write("""
#This dashboard provides tools to analyze your EV charging data and predict future energy demand. 
#Upload your data below to get started.
#""")

# Wrap the upload section in a styled div
st.markdown('<div class="upload-section">', unsafe_allow_html=True)
st.subheader("📤 Upload Your Data")
uploaded_file = st.file_uploader("Upload a CSV file", type=['csv'])
st.markdown('</div>', unsafe_allow_html=True)

# File upload section
# st.subheader("Upload Your Data")
# uploaded_file = st.file_uploader("Upload a CSV file", type=['csv'])

if uploaded_file is not None:
    try:
        # Read the CSV file with date parsing
        input_df = pd.read_csv(uploaded_file)
        
        # Convert date column to datetime if it exists
        if date_column in input_df.columns:
            input_df[date_column] = pd.to_datetime(input_df[date_column])
        
        # Create tabs for different analyses
        tab1, tab2 = st.tabs(["Data Analysis", "Predictions"])

        st.markdown("""
            <style>
            .stTabs [data-baseweb="tab-list"] {
                gap: 24px;
            }
            .stTabs [data-baseweb="tab"] {
                color: #707070;
                padding: 0 24px;
            }
            .stTabs [aria-selected="true"] {
                color: #1E88E5;
                border-bottom-color: #1E88E5;
            }
            </style>
        """, unsafe_allow_html=True)
        
        with tab1:
            st.subheader("📊 Data Analysis")
            
            # Data preview
            st.write("### Data Preview")
            st.dataframe(input_df.head())
            
            # Basic statistics
            st.write("### Statistical Summary")
            st.dataframe(input_df.describe())
            
            # Feature correlations
            st.write("### Feature Correlations")
            correlation_matrix = input_df.select_dtypes(include=[np.number]).corr()
            fig_corr = px.imshow(correlation_matrix,
                               labels=dict(color="Correlation"),
                               color_continuous_scale="RdBu")
            st.plotly_chart(fig_corr, use_container_width=True)
            
            # Time series analysis
            if date_column in input_df.columns:
                st.write("### Time Series Analysis")
                numeric_cols = input_df.select_dtypes(include=[np.number]).columns
                selected_features = st.multiselect(
                    "Select features to plot",
                    options=numeric_cols,
                    default=[target_variable] if target_variable in numeric_cols else [numeric_cols[0]]
                )
                
                #fig_ts = px.line(input_df, x=date_column, y=selected_features)
                #st.plotly_chart(fig_ts, use_container_width=True)

                fig_ts = px.line(input_df, 
                    x=date_column, 
                    y=selected_features,
                    template='plotly_white',  # Cleaner template
                    labels={'value': 'Energy Consumption (kWh)', 
                            'date': 'Date',
                            'variable': 'Features'},
                    title='Time Series Analysis',
                    color_discrete_map={
                        'total_energy': '#1E88E5',
                        'total_ghg_savings': '#FFC107',
                        'total_charging_sec': '#4CAF50',
                        '7_rolling_avg': '#FF5252',
                        '30_rolling_avg': '#9C27B0'
                    }
                )
    
                # Update layout to match prediction plot style
                fig_ts.update_layout(
                    plot_bgcolor='white',
                    title_x=0.5,  # Center the title
                    title_font_size=20,
                    legend=dict(
                        orientation="h",
                        yanchor="bottom",
                        y=1.02,
                        xanchor="right",
                        x=1
                    ),
                    margin=dict(t=100),  # Add more top margin for the legend
                    hovermode='x unified'
                )
    
                # Update axes to match prediction plot style
                fig_ts.update_xaxes(showgrid=True, gridwidth=1, gridcolor='#E0E0E0')
                fig_ts.update_yaxes(showgrid=True, gridwidth=1, gridcolor='#E0E0E0')
    
                st.plotly_chart(fig_ts, use_container_width=True)
        
        with tab2:
            st.subheader("🔮 Predictions")
    
            if st.button("Generate Predictions"):
                # Extract features for prediction
                features_df = input_df[features]
        
                # Fit and transform the input data
                X_scaler.fit(features_df)  # Add this line to fit the scaler
                input_scaled = X_scaler.transform(features_df)
        
                # Fit and transform the target variable
                y_data = input_df[target_variable].values.reshape(-1, 1)
                y_scaler.fit(y_data)  # Add this line to fit the scaler
        
                # Reshape for LSTM
                input_lstm = input_scaled.reshape((input_scaled.shape[0], 1, input_scaled.shape[1]))
                
                # Make predictions
                predictions = model.predict(input_lstm)
                final_predictions = y_scaler.inverse_transform(predictions)
                
                # Create results dataframe
                results_df = pd.DataFrame({
                    'Date': input_df[date_column],
                    'Actual': input_df[target_variable].values,
                    'Predicted': final_predictions.flatten()
                })

                # Create results dataframe and store in session state
                st.session_state.results_df = pd.DataFrame({
                    'Date': input_df[date_column],
                    'Actual': input_df[target_variable].values,
                    'Predicted': final_predictions.flatten()
                })
                # Display results
                st.write("### Prediction Results")
                st.dataframe(results_df)
                
                # Download button for predictions
                csv = results_df.to_csv(index=False)
                st.download_button(
                    label="Download Predictions",
                    data=csv,
                    file_name="predictions.csv",
                    mime="text/csv"
                )
                
                # Visualization
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("### Time Series Comparison")
                    fig_time = px.line(results_df,
                                     x='Date',
                                     y=['Actual', 'Predicted'],
                                     title='Actual vs Predicted Values Over Time')
                    st.plotly_chart(fig_time, use_container_width=True)
                
                with col2:
                    st.write("### Prediction Correlation")
                    fig_scatter = px.scatter(results_df,
                                           x='Actual',
                                           y='Predicted',
                                           title='Actual vs Predicted Values')
                    
                    # Add perfect prediction line
                    fig_scatter.add_trace(
                        go.Scatter(
                            x=[results_df['Actual'].min(), results_df['Actual'].max()],
                            y=[results_df['Actual'].min(), results_df['Actual'].max()],
                            mode='lines',
                            name='Perfect Prediction',
                            line=dict(dash='dash', color='red')
                        )
                    )
                    st.plotly_chart(fig_scatter, use_container_width=True)
        
            #with tab3:
                #st.subheader("📈 Model Performance")
    
                #if 'results_df' in st.session_state:
                    # Calculate performance metrics
                    #mae = np.mean(np.abs(st.session_state.results_df['Actual'] - st.session_state.results_df['Predicted']))
                    #rmse = np.sqrt(np.mean((st.session_state.results_df['Actual'] - st.session_state.results_df['Predicted'])**2))
                    #r2 = 1 - (np.sum((st.session_state.results_df['Actual'] - st.session_state.results_df['Predicted'])**2) / 
                         #np.sum((st.session_state.results_df['Actual'] - np.mean(st.session_state.results_df['Actual']))**2))
        
                    # Display metrics
                    #col1, col2, col3 = st.columns(3)
                    #with col1:
                        #st.metric("Mean Absolute Error", f"{mae:.3f}")
                    #with col2:
                        #st.metric("Root Mean Squared Error", f"{rmse:.3f}")
                    #with col3:
                        #st.metric("R² Score", f"{r2:.3f}")
        
                    # Additional performance visualizations
                    #st.write("### Residual Analysis")
                    #st.session_state.results_df['Residuals'] = st.session_state.results_df['Actual'] - st.session_state.results_df['Predicted']
        
                    #fig_residuals = px.scatter(
                        #st.session_state.results_df,
                        #x='Predicted',
                        #y='Residuals',
                        #title='Residual Plot'
                    #)
                    #fig_residuals.add_hline(y=0, line_dash="dash", line_color="red")
                    #st.plotly_chart(fig_residuals, use_container_width=True)
                #else:
                    #st.info("Generate predictions first to see model performance metrics.")

    except Exception as e:
        st.error(f"Error processing the file: {str(e)}")
        st.write("Please make sure your CSV file has the following columns:")
        st.write(f"Date column: {date_column}")
        st.write("Feature columns:", ", ".join(features))
        st.write(f"Target column: {target_variable}")

else:
    # Show template information when no file is uploaded
    st.info("👆 Upload a CSV file to get started")
    
    st.markdown("### CSV Template")
    st.write("Your CSV should have the following columns:")
    
    # Create sample template
    sample_df = pd.DataFrame(columns=[date_column] + features + [target_variable])
    st.dataframe(sample_df)
    
    # Download template button
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

# Footer
st.markdown("---")
st.markdown("Created with ❤️ using Streamlit")
