import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from tensorflow.keras.models import load_model
import joblib

def setup_page():
    """Configure the page settings and apply custom styling."""
    st.set_page_config(
        page_title="EV Energy Dashboard",
        page_icon="📈",
        layout="wide"
    )
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)
    st.markdown('<div class="dashboard-title">EV Energy Demand Analysis & Prediction Dashboard</div>', 
                unsafe_allow_html=True)
    st.markdown('<div class="dashboard-subtitle">This dashboard provides tools to analyze your EV charging data and predict future energy demand.</div>', 
                unsafe_allow_html=True)

# Constants
DATE_COLUMN = 'date'
FEATURES = ['total_ghg_savings', 'total_charging_sec', '7_rolling_avg', '30_rolling_avg',
           'lag_1', 'lag_2', 'lag_3', 'lag_7', 'day_of_week', 'is_weekend', 'month']
TARGET_VARIABLE = 'total_energy'

st.markdown("""
    <style>
        .stApp {
            background-color: #f0f2f6;
            font-family: 'Roboto', sans-serif;
            }

        .dashboard-title {
            font-size: 2.5rem;
            font-weight: 700;
            color: #1a237e;
            text-align: center;
            padding: 1.5rem 0;
            background-color: white;
            border-radius: 10px;
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
            margin-bottom: 2rem;
            }

        .dashboard-subtitle {
            font-size: 1.1rem;
            color: #424242;
            text-align: center;
            margin-bottom: 2rem;
            }

        h2, h3 {
            font-family: 'Roboto', sans-serif;
            color: #1a237e;
            font-weight: 500;
            padding: 1rem 0;
            }

        .plot-container {
            background-color: white;
            padding: 1.5rem;
            border-radius: 10px;
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
            margin: 1rem 0;
            }

        div[data-testid="metric-container"] {
            background-color: white;
            border: 1px solid #e0e0e0;
            padding: 1rem;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
            }

        [data-testid="stFileUploader"] {
            background-color: white;
            padding: 1.5rem;
            border-radius: 10px;
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
            }

        .stTabs [data-baseweb="tab-list"] {
            gap: 24px;
            background-color: white;
            padding: 0.5rem 1rem;
            border-radius: 10px;
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
            }

        .stTabs [data-baseweb="tab"] {
            color: #424242;
            font-weight: 500;
            }

        .stTabs [aria-selected="true"] {
            color: #1a237e;
            border-bottom-color: #1a237e;
            }
    </style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_saved_model():
    """Load the saved model and scalers."""
    model = load_model('lstm_model.h5')
    X_scaler = joblib.load('X_scaler.pkl')
    y_scaler = joblib.load('y_scaler.pkl')
    return model, X_scaler, y_scaler

def create_time_series_plot(df, date_column, selected_features):
    """Create a time series plot with the selected features."""
    fig = px.line(df, 
        x=date_column, 
        y=selected_features,
        template='plotly_white',
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
    
    fig.update_layout(
        plot_bgcolor='white',
        title_x=0.5,
        title_font_size=20,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        margin=dict(t=100),
        hovermode='x unified'
    )
    
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='#E0E0E0')
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='#E0E0E0')
    
    return fig

def create_prediction_plots(results_df):
    """Create time series comparison and prediction correlation plots."""
    # Time Series Comparison Plot
    fig_time = px.line(results_df,
        x='Date',
        y=['Actual', 'Predicted'],
        template='plotly_white',
        labels={'value': 'Energy Consumption (kWh)', 
                'Date': 'Date',
                'variable': 'Type'},
        color_discrete_map={'Actual': '#1E88E5', 'Predicted': '#FFC107'}
    )
    
    fig_time.update_layout(
        title={
            'text': 'Energy Consumption: Actual vs Predicted',
            'y': 0.95,
            'x': 0.5,
            'xanchor': 'center',
            'yanchor': 'top',
            'font': dict(size=20, color='#1a237e')
        },
        plot_bgcolor='white',
        paper_bgcolor='white',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5
        ),
        margin=dict(l=60, r=30, t=100, b=50),
        hovermode='x unified'
    )
    
    # Prediction Correlation Plot
    fig_scatter = px.scatter(results_df,
        x='Actual',
        y='Predicted',
        template='plotly_white',
        labels={'Actual': 'Actual Energy Consumption (kWh)', 
                'Predicted': 'Predicted Energy Consumption (kWh)'}
    )
    
    # Add perfect prediction line
    fig_scatter.add_trace(
        go.Scatter(
            x=[results_df['Actual'].min(), results_df['Actual'].max()],
            y=[results_df['Actual'].min(), results_df['Actual'].max()],
            mode='lines',
            name='Perfect Prediction',
            line=dict(dash='dash', color='#FF5252', width=2)
        )
    )
    
    fig_scatter.update_layout(
        title={
            'text': 'Prediction Accuracy Analysis',
            'y': 0.95,
            'x': 0.5,
            'xanchor': 'center',
            'yanchor': 'top',
            'font': dict(size=20, color='#1a237e')
        },
        plot_bgcolor='white',
        paper_bgcolor='white',
        showlegend=True
    )
    
    return fig_time, fig_scatter

def main():
    setup_page()
    model, X_scaler, y_scaler = load_saved_model()
    
    st.subheader("Upload Data")
    uploaded_file = st.file_uploader("Upload a CSV file", type=['csv'])
    
    if uploaded_file is not None:
        try:
            input_df = pd.read_csv(uploaded_file)
            if DATE_COLUMN in input_df.columns:
                input_df[DATE_COLUMN] = pd.to_datetime(input_df[DATE_COLUMN])
            
            tab1, tab2 = st.tabs(["Data Analysis", "Predictions"])
            
            with tab1:
                st.subheader("📊 Data Analysis")
                
                # Data Preview
                st.write("### Data Preview")
                st.dataframe(input_df.head())
                
                # Statistical Summary
                st.write("### Statistical Summary")
                st.dataframe(input_df.describe())
                
                # Feature Correlations
                st.write("### Feature Correlations")
                correlation_matrix = input_df.select_dtypes(include=[np.number]).corr()
                fig_corr = px.imshow(correlation_matrix,
                                   labels=dict(color="Correlation"),
                                   color_continuous_scale="RdBu")
                st.plotly_chart(fig_corr, use_container_width=True)
                
                # Time Series Analysis
                if DATE_COLUMN in input_df.columns:
                    st.write("### Time Series Analysis")
                    numeric_cols = input_df.select_dtypes(include=[np.number]).columns
                    selected_features = st.multiselect(
                        "Select features to plot",
                        options=numeric_cols,
                        default=[TARGET_VARIABLE] if TARGET_VARIABLE in numeric_cols else [numeric_cols[0]]
                    )
                    
                    fig_ts = create_time_series_plot(input_df, DATE_COLUMN, selected_features)
                    st.plotly_chart(fig_ts, use_container_width=True)
            
            with tab2:
                st.subheader("🔮 Predictions")
                
                if st.button("Generate Predictions"):
                    features_df = input_df[FEATURES]
                    X_scaler.fit(features_df)
                    input_scaled = X_scaler.transform(features_df)
                    
                    y_data = input_df[TARGET_VARIABLE].values.reshape(-1, 1)
                    y_scaler.fit(y_data)
                    
                    input_lstm = input_scaled.reshape((input_scaled.shape[0], 1, input_scaled.shape[1]))
                    predictions = model.predict(input_lstm)
                    final_predictions = y_scaler.inverse_transform(predictions)
                    
                    results_df = pd.DataFrame({
                        'Date': input_df[DATE_COLUMN],
                        'Actual': input_df[TARGET_VARIABLE].values,
                        'Predicted': final_predictions.flatten()
                    })
                    
                    st.write("### Prediction Results")
                    st.dataframe(results_df)
                    
                    # Download predictions
                    csv = results_df.to_csv(index=False)
                    st.download_button(
                        label="Download Predictions",
                        data=csv,
                        file_name="predictions.csv",
                        mime="text/csv"
                    )
                    
                    # Create and display prediction plots
                    fig_time, fig_scatter = create_prediction_plots(results_df)
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.plotly_chart(fig_time, use_container_width=True)
                    with col2:
                        st.plotly_chart(fig_scatter, use_container_width=True)
        
        except Exception as e:
            st.error(f"Error processing the file: {str(e)}")
            st.write("Please make sure your CSV file has the following columns:")
            st.write(f"Date column: {DATE_COLUMN}")
            st.write("Feature columns:", ", ".join(FEATURES))
            st.write(f"Target column: {TARGET_VARIABLE}")
    
    else:
        st.info("👆 Upload a CSV file to get started")
        st.markdown("### CSV Template")
        st.write("Your CSV should have the following columns:")
        
        sample_df = pd.DataFrame(columns=[DATE_COLUMN] + FEATURES + [TARGET_VARIABLE])
        st.dataframe(sample_df)
        
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

if __name__ == "__main__":
    main()
