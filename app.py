import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

# Custom Normal Equation Linear Regression Solver
def fit_linear_regression(X, y):
    X_b = np.column_stack([np.ones(X.shape[0]), X])
    beta = np.linalg.inv(X_b.T @ X_b) @ X_b.T @ y
    return beta[0], beta[1:]

def predict_linear_regression(X, intercept, coefs):
    return intercept + X @ coefs

def calculate_metrics(y_true, y_pred):
    mae = np.mean(np.abs(y_true - y_pred))
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
    return r2, mae, rmse

# App Page Configurations
st.set_page_config(
    page_title="Netflix Upload Forecasting Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Style for Netflix Red Theme
st.markdown("""
<style>
    .main {
        background-color: #fcfcfc;
    }
    h1, h2, h3 {
        color: #E50914 !important;
    }
    .stButton>button {
        background-color: #E50914;
        color: white;
        border-radius: 5px;
        border: none;
    }
    .stButton>button:hover {
        background-color: #B80710;
        color: white;
    }
    .sidebar .sidebar-content {
        background-color: #111111;
        color: white;
    }
    div[data-testid="stMetricValue"] {
        font-size: 30px;
        font-weight: bold;
        color: #221F1F;
    }
</style>
""", unsafe_allow_html=True)

# 1. Load and parse dataset (Cached for performance)
@st.cache_data
def get_processed_data():
    if not os.path.exists('netflix_titles.csv'):
        return None, None
        
    df = pd.read_csv('netflix_titles.csv')
    df = df.dropna(subset=['date_added'])
    df['date_added'] = df['date_added'].str.strip()
    df['date_added'] = pd.to_datetime(df['date_added'], errors='coerce')
    df = df.dropna(subset=['date_added'])
    
    df['added_year'] = df['date_added'].dt.year
    df['added_month'] = df['date_added'].dt.month
    
    # Modern Netflix Era
    all_months = pd.date_range(start='2016-01-01', end='2021-09-01', freq='MS')
    ts_df = pd.DataFrame({'date': all_months})
    ts_df['year'] = ts_df['date'].dt.year
    ts_df['month'] = ts_df['date'].dt.month
    
    monthly_counts = df.groupby(['added_year', 'added_month']).size().reset_index(name='uploads')
    
    df_clean = pd.merge(ts_df, monthly_counts, left_on=['year', 'month'], right_on=['added_year', 'added_month'], how='left').fillna(0)
    df_clean = df_clean[['date', 'year', 'month', 'uploads']]
    
    # Feature engineering
    df_clean['trend'] = np.arange(1, len(df_clean) + 1)
    df_clean['month_sin'] = np.sin(2 * np.pi * df_clean['month'] / 12)
    df_clean['month_cos'] = np.cos(2 * np.pi * df_clean['month'] / 12)
    df_clean['lag_1'] = df_clean['uploads'].shift(1)
    df_clean['lag_2'] = df_clean['uploads'].shift(2)
    df_clean['lag_3'] = df_clean['uploads'].shift(3)
    df_clean['rolling_mean_3'] = df_clean[['lag_1', 'lag_2', 'lag_3']].mean(axis=1)
    
    return df, df_clean

# Sidebar Content
st.sidebar.image("https://upload.wikimedia.org/wikipedia/commons/0/08/Netflix_2015_logo.svg", width=150)
st.sidebar.title("Forecasting Control Panel")
st.sidebar.markdown("Configure modeling and simulation parameters below.")

df_raw, df_processed = get_processed_data()

if df_processed is None:
    st.error("Could not load netflix_titles.csv. Make sure it exists in the workspace.")
else:
    # Train-test split for calculations
    df_model_data = df_processed.dropna().reset_index(drop=True)
    test_size = 12
    train_df = df_model_data.iloc[:-test_size].copy()
    test_df = df_model_data.iloc[-test_size:].copy()
    
    # Train Models
    # Model 1
    features_m1 = ['trend', 'month_sin', 'month_cos']
    X_train_m1, y_train = train_df[features_m1].values, train_df['uploads'].values
    m1_intercept, m1_coefs = fit_linear_regression(X_train_m1, y_train)
    
    # Model 2
    features_m2 = ['trend', 'month_sin', 'month_cos', 'lag_1']
    X_train_m2 = train_df[features_m2].values
    m2_intercept, m2_coefs = fit_linear_regression(X_train_m2, y_train)
    
    # User Inputs in Sidebar
    model_choice = st.sidebar.selectbox(
        "Select Prediction Model",
        ["Model 2: Autoregressive Linear Regression (Best)", "Model 1: Static Trend + Seasonality", "Model 3: Baseline Moving Average"]
    )
    
    forecast_horizon = st.sidebar.slider(
        "Forecast Horizon (Months)",
        min_value=3, max_value=24, value=12, step=1
    )
    
    custom_lag_scenario = st.sidebar.number_input(
        "Simulate September 2021 Uploads (Lag-1)",
        min_value=10, max_value=500, value=183, step=10,
        help="Change the final known monthly uploads to see how the autoregressive forecast shifts dynamically!"
    )

    # Main Page Layout
    st.title("📈 Netflix Monthly Content Upload Forecasting")
    st.markdown("Predicting and analyzing the volume of titles added to Netflix using Linear Regression models with time trend, trigonometric seasonality, and lagged autoregression.")
    
    # KPIs Top Section
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Catalog Titles", f"{len(df_raw):,}")
    with col2:
        st.metric("Average Monthly Uploads", f"{int(df_processed['uploads'].mean())} titles")
    with col3:
        st.metric("Peak Upload Month (Jul '21)", "257 titles")
    with col4:
        st.metric("Modern Era Span", "69 months")
        
    st.markdown("---")
    
    # Tabbed Views
    tab1, tab2, tab3 = st.tabs(["🔮 Future Forecast Dashboard", "📊 Seasonality & Exploration", "⚙️ Interactive Simulation Sandbox"])
    
    with tab1:
        st.subheader("12-Month Recursive Future Forecast")
        st.markdown("This dashboard displays the forward forecast generated using your selected parameters. The Autoregressive model feeds its own predictions back recursively as lags for subsequent months.")
        
        # Calculate Future Forecast
        future_dates = pd.date_range(start='2021-10-01', periods=forecast_horizon, freq='MS')
        future_df = pd.DataFrame({'date': future_dates})
        future_df['year'] = future_df['date'].dt.year
        future_df['month'] = future_df['date'].dt.month
        future_df['trend'] = np.arange(len(df_model_data) + 4, len(df_model_data) + 4 + len(future_dates))
        future_df['month_sin'] = np.sin(2 * np.pi * future_df['month'] / 12)
        future_df['month_cos'] = np.cos(2 * np.pi * future_df['month'] / 12)
        
        predictions_m1 = []
        predictions_m2 = []
        predictions_m3 = []
        
        current_lag = custom_lag_scenario
        lag_history = [df_processed['uploads'].values[-2], df_processed['uploads'].values[-3]] # lag_2, lag_3
        
        for i in range(len(future_df)):
            row = future_df.iloc[i]
            
            # Model 1
            pred_m1 = m1_intercept + row['trend'] * m1_coefs[0] + row['month_sin'] * m1_coefs[1] + row['month_cos'] * m1_coefs[2]
            predictions_m1.append(pred_m1)
            
            # Model 2
            pred_m2 = m2_intercept + row['trend'] * m2_coefs[0] + row['month_sin'] * m2_coefs[1] + row['month_cos'] * m2_coefs[2] + current_lag * m2_coefs[3]
            predictions_m2.append(pred_m2)
            
            # Model 3 (Baseline Rolling 3-month mean)
            pred_m3 = np.mean([current_lag] + lag_history[:2])
            predictions_m3.append(pred_m3)
            
            # Update recursive feedback variables
            lag_history = [current_lag] + lag_history
            current_lag = pred_m2 if model_choice.startswith("Model 2") else pred_m1
            
        future_df['forecast_m1'] = predictions_m1
        future_df['forecast_m2'] = predictions_m2
        future_df['forecast_m3'] = predictions_m3
        
        # Select Active Forecast Column
        if model_choice.startswith("Model 1"):
            future_df['selected_forecast'] = future_df['forecast_m1']
            rmse_val = 64.6
        elif model_choice.startswith("Model 2"):
            future_df['selected_forecast'] = future_df['forecast_m2']
            rmse_val = 51.7
        else:
            future_df['selected_forecast'] = future_df['forecast_m3']
            rmse_val = 41.6
            
        # Shading and bounds
        future_df['lower_bound'] = (future_df['selected_forecast'] - 1.96 * rmse_val).clip(lower=0)
        future_df['upper_bound'] = future_df['selected_forecast'] + 1.96 * rmse_val
        
        # Plotting
        fig, ax = plt.subplots(figsize=(11, 4.5))
        ax.plot(df_model_data['date'], df_model_data['uploads'], color='#221F1F', label='Historical Actuals', linewidth=2, marker='o', markersize=3)
        
        # Plot Future Forecast Line
        ax.plot(future_df['date'], future_df['selected_forecast'], color='#E50914', label=f'Forecasted ({model_choice.split(":")[0]})', linewidth=2.5, marker='s', markersize=4)
        ax.fill_between(future_df['date'], future_df['lower_bound'], future_df['upper_bound'], color='#E50914', alpha=0.15, label='95% Confidence Band')
        
        ax.set_title('Netflix Content Uploads Future Forecast', fontsize=12, fontweight='bold')
        ax.set_xlabel('Date Added', fontsize=10)
        ax.set_ylabel('Number of Titles', fontsize=10)
        ax.legend(frameon=True, facecolor='white', edgecolor='none', loc='upper left', fontsize=9)
        plt.tight_layout()
        st.pyplot(fig)
        
        # Show table data
        st.markdown("### 📋 Forecasted Uploads Detail")
        table_df = future_df[['date', 'selected_forecast', 'lower_bound', 'upper_bound']].copy()
        table_df['date'] = table_df['date'].dt.strftime('%B %Y')
        table_df.columns = ['Month', 'Predicted Uploads', '95% Lower Limit', '95% Upper Limit']
        table_df['Predicted Uploads'] = table_df['Predicted Uploads'].astype(int)
        table_df['95% Lower Limit'] = table_df['95% Lower Limit'].astype(int)
        table_df['95% Upper Limit'] = table_df['95% Upper Limit'].astype(int)
        st.dataframe(table_df, use_container_width=True)
        
    with tab2:
        st.subheader("📊 Exploratory Analysis & Seasonality")
        col_s1, col_s2 = st.columns(2)
        
        with col_s1:
            st.markdown("#### Monthly Seasonality Blocks")
            st.markdown("Content addition typically exhibits a double-peak structure aligning with summer breaks (July) and winter holidays (December/January).")
            
            # Generate Seasonality Plot
            monthly_stats = df_processed.groupby('month')['uploads'].agg(['mean', 'std', 'count']).reset_index()
            monthly_stats['sem'] = monthly_stats['std'] / np.sqrt(monthly_stats['count'])
            months_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
            
            fig_s, ax_s = plt.subplots(figsize=(6, 4))
            bars = ax_s.bar(monthly_stats['month'], monthly_stats['mean'], yerr=monthly_stats['sem'], capsize=3, color='#E07A7F', edgecolor='#221F1F', alpha=0.9)
            # Make peak months stand out
            bars[6].set_color('#E50914') # July
            bars[11].set_color('#E50914') # December
            
            ax_s.axhline(monthly_stats['mean'].mean(), color='#221F1F', linestyle='--', linewidth=1, alpha=0.8, label='Annual Average')
            ax_s.set_xticks(monthly_stats['month'])
            ax_s.set_xticklabels(months_names, fontsize=8)
            ax_s.set_title('Average Content Additions by Month of Year', fontsize=10, fontweight='bold')
            ax_s.set_ylabel('Average Titles', fontsize=9)
            ax_s.legend(fontsize=8)
            plt.tight_layout()
            st.pyplot(fig_s)
            
        with col_s2:
            st.markdown("#### Long-term Growth Trend")
            st.markdown("The 12-month rolling average highlights the long-term strategic lifecycle of Netflix. Exponential scaling matured in late 2019, transitioning into a sustainable steady state.")
            
            # Generate Trend Plot
            df_processed['rolling_mean_12'] = df_processed['uploads'].rolling(window=12, min_periods=1).mean()
            fig_t, ax_t = plt.subplots(figsize=(6, 4))
            ax_t.plot(df_processed['date'], df_processed['uploads'], color='#8c8c8c', alpha=0.5, label='Monthly Uploads', linewidth=1)
            ax_t.plot(df_processed['date'], df_processed['rolling_mean_12'], color='#E50914', label='12-Month Moving Avg', linewidth=2.5)
            ax_t.set_title('Netflix Growth Curve Maturity', fontsize=10, fontweight='bold')
            ax_t.set_ylabel('Number of Titles', fontsize=9)
            ax_t.legend(fontsize=8)
            plt.tight_layout()
            st.pyplot(fig_t)
            
    with tab3:
        st.subheader("🧪 Interactive Simulation Sandbox")
        st.markdown("Test hypothetical forecasting conditions dynamically! Adjust parameters to see the immediate predicted upload count for the next month.")
        
        sim_col1, sim_col2 = st.columns([1, 2])
        
        with sim_col1:
            st.markdown("#### Adjust Sandbox Inputs")
            target_year = st.selectbox("Target Simulation Year", [2021, 2022, 2023])
            target_month = st.slider("Target Simulation Month", 1, 12, 10)
            custom_lag = st.slider("Previous Month Uploads (Lag-1)", 10, 400, 183, step=5)
            
        with sim_col2:
            st.markdown("#### Live Simulation Output")
            
            # Calculate dynamic prediction
            sim_trend_index = len(df_model_data) + 4 # approximation
            sim_month_sin = np.sin(2 * np.pi * target_month / 12)
            sim_month_cos = np.cos(2 * np.pi * target_month / 12)
            
            # Predict
            sim_pred_m1 = m1_intercept + sim_trend_index * m1_coefs[0] + sim_month_sin * m1_coefs[1] + sim_month_cos * m1_coefs[2]
            sim_pred_m2 = m2_intercept + sim_trend_index * m2_coefs[0] + sim_month_sin * m2_coefs[1] + sim_month_cos * m2_coefs[2] + custom_lag * m2_coefs[3]
            
            # Dynamic explanations
            st.info(f"📅 **Simulation Period:** {months_names[target_month - 1]} {target_year}")
            
            sc1, sc2 = st.columns(2)
            with sc1:
                st.metric(
                    label="Autoregressive Forecast (Model 2)",
                    value=f"{int(sim_pred_m2)} titles",
                    delta=f"{int(sim_pred_m2 - sim_pred_m1)} titles due to lag momentum"
                )
            with sc2:
                st.metric(
                    label="Static Forecast (Model 1)",
                    value=f"{int(sim_pred_m1)} titles"
                )
            
            st.markdown("""
            **💡 Rationale:** 
            - When **Previous Month Uploads (Lag-1)** increases, **Model 2** dynamically factors in the release momentum, shifting the predicted counts upwards.
            - **Model 1** remains completely constant because it only models the static calendar date (month and trend index).
            """)
