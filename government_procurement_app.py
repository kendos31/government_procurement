

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from sklearn.ensemble import RandomForestRegressor, IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score
import joblib
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Government Procurement Analytics",
    page_icon="kendos.png",
    layout="wide",
    initial_sidebar_state="expanded"
)

class ProcurementAnalytics:
    def __init__(self):
        self.forecast_model = None
        self.anomaly_model = None
        self.scaler = None
        self.data = None
        self.monthly_data = None

    def load_and_preprocess(self, df):
        """Load and preprocess procurement data"""
        # Try to detect date format automatically
        date_columns = ['award_date', 'date', 'Date', 'award_date']
        for col in date_columns:
            if col in df.columns:
                try:
                    df['award_date'] = pd.to_datetime(df[col], errors='coerce')
                    break
                except:
                    continue

        if 'award_date' not in df.columns:
            st.error("Could not find date column. Please ensure your data has an 'award_date' column or similar.")
            return None

        # Try to detect amount column
        amount_columns = ['awarded_amt', 'amount', 'Amount', 'value', 'award_amount']
        for col in amount_columns:
            if col in df.columns and col != 'award_date':
                df['awarded_amt'] = pd.to_numeric(df[col], errors='coerce').fillna(0)
                break

        if 'awarded_amt' not in df.columns:
            st.error("Could not find amount column. Please ensure your data has an 'awarded_amt' column or similar.")
            return None

        # Filter valid transactions
        df = df[df['awarded_amt'] > 0]

        # Ensure required columns exist
        required_columns = ['agency', 'supplier_name']
        for col in required_columns:
            if col not in df.columns:
                df[col] = f"Unknown {col}"
        df['anomaly_score'] = 0
        df['is_anomaly'] = 0
        self.data = df
        return df

    def create_features(self, df):
        """Create features for modeling"""
        # Time-based features
        df['year'] = df['award_date'].dt.year
        df['month'] = df['award_date'].dt.month
        df['quarter'] = df['award_date'].dt.quarter
        df['day_of_week'] = df['award_date'].dt.dayofweek

        return df

    def prepare_forecast_data(self, df):
        """Prepare time series data for forecasting"""
        # Aggregate monthly
        monthly = df.groupby(df['award_date'].dt.to_period('M')).agg({
            'awarded_amt': ['sum', 'mean', 'count']
        }).reset_index()

        monthly.columns = ['month', 'total_spending', 'avg_spending', 'transaction_count']
        monthly['month'] = monthly['month'].dt.to_timestamp()

        # Add lag features
        for lag in [1, 2, 3, 6, 12]:
            monthly[f'spending_lag_{lag}'] = monthly['total_spending'].shift(lag)

        # Add rolling statistics
        monthly['rolling_mean_3'] = monthly['total_spending'].rolling(3).mean()
        monthly['rolling_std_3'] = monthly['total_spending'].rolling(3).std()
        monthly['rolling_mean_6'] = monthly['total_spending'].rolling(6).mean()
        monthly['rolling_std_6'] = monthly['total_spending'].rolling(6).std()

        # Add seasonal features
        monthly['month_sin'] = np.sin(2 * np.pi * monthly['month'].dt.month / 12)
        monthly['month_cos'] = np.cos(2 * np.pi * monthly['month'].dt.month / 12)

        self.monthly_data = monthly.dropna()
        return self.monthly_data

    def train_forecast_model(self, monthly_data):
        """Train spending forecast model"""
        features = ['avg_spending', 'transaction_count',
                   'spending_lag_1', 'spending_lag_2', 'spending_lag_3',
                   'spending_lag_6', 'spending_lag_12',
                   'rolling_mean_3', 'rolling_std_3',
                   'rolling_mean_6', 'rolling_std_6',
                   'month_sin', 'month_cos']

        # Use only features that exist in the data
        available_features = [f for f in features if f in monthly_data.columns]

        target = 'total_spending'

        X = monthly_data[available_features]
        y = monthly_data[target]

        # Scale features
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)

        # Train model
        self.forecast_model = RandomForestRegressor(
            n_estimators=200,
            random_state=42,
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=2
        )
        self.forecast_model.fit(X_scaled, y)

        # Evaluate
        predictions = self.forecast_model.predict(X_scaled)
        mae = mean_absolute_error(y, predictions)
        r2 = r2_score(y, predictions)

        return mae, r2, predictions, available_features

    def predict_future_months(self, n_months=12):
        """Predict future monthly spending"""
        if self.forecast_model is None or self.scaler is None or self.monthly_data is None:
            return None

        # Get the last n months of data for creating future predictions
        last_data = self.monthly_data.copy().sort_values('month')
        last_row = last_data.iloc[-1].copy()

        predictions = []
        feature_names = [col for col in last_data.columns if col not in ['month', 'total_spending']]

        for i in range(1, n_months + 1):
            # Create a new month
            future_month = last_row['month'] + pd.DateOffset(months=i)
            new_row = last_row.copy()
            new_row['month'] = future_month

            # Update lag features (shift existing values)
            for lag in [1, 2, 3, 6, 12]:
                lag_col = f'spending_lag_{lag}'
                if lag == 1 and predictions:
                    new_row[lag_col] = predictions[-1]['predicted_spending']
                elif lag > 1 and len(predictions) >= lag - 1:
                    new_row[lag_col] = predictions[-(lag - 1)]['predicted_spending']
                elif lag_col in last_data.columns:
                    # Shift existing data
                    if len(last_data) >= lag:
                        new_row[lag_col] = last_data['total_spending'].iloc[-lag]

            # Update rolling statistics
            recent_values = list(last_data['total_spending'].tail(6))
            if predictions:
                recent_values.extend([p['predicted_spending'] for p in predictions[-6:]])

            if len(recent_values) >= 3:
                new_row['rolling_mean_3'] = np.mean(recent_values[-3:])
                new_row['rolling_std_3'] = np.std(recent_values[-3:])

            if len(recent_values) >= 6:
                new_row['rolling_mean_6'] = np.mean(recent_values[-6:])
                new_row['rolling_std_6'] = np.std(recent_values[-6:])

            # Update seasonal features
            new_row['month_sin'] = np.sin(2 * np.pi * future_month.month / 12)
            new_row['month_cos'] = np.cos(2 * np.pi * future_month.month / 12)

            # Prepare features for prediction
            features_for_pred = self.monthly_data.columns.tolist()
            features_for_pred.remove('month')
            features_for_pred.remove('total_spending')

            X_future = new_row[features_for_pred].values.reshape(1, -1)
            X_future_scaled = self.scaler.transform(X_future)

            # Make prediction
            predicted_spending = max(0, self.forecast_model.predict(X_future_scaled)[0])

            predictions.append({
                'month': future_month,
                'predicted_spending': predicted_spending,
                'confidence_interval_low': predicted_spending * 0.8,  # 80% confidence
                'confidence_interval_high': predicted_spending * 1.2  # 80% confidence
            })

            # Update last_row for next iteration
            last_row = new_row.copy()
            last_row['total_spending'] = predicted_spending

        return pd.DataFrame(predictions)

    def predict_yearly_summary(self, n_years=3):
        """Predict yearly spending summary"""
        monthly_predictions = self.predict_future_months(n_months=n_years * 12)

        if monthly_predictions is None:
            return None

        # Group by year
        monthly_predictions['year'] = monthly_predictions['month'].dt.year
        yearly_predictions = monthly_predictions.groupby('year').agg({
            'predicted_spending': 'sum',
            'confidence_interval_low': 'sum',
            'confidence_interval_high': 'sum'
        }).reset_index()

        return yearly_predictions

    def detect_anomalies(self, df, contamination=0.05):
        """Detect anomalous procurement transactions"""
        # Calculate agency statistics
        agency_stats = df.groupby('agency')['awarded_amt'].agg(['mean', 'std']).reset_index()
        agency_stats.columns = ['agency', 'agency_mean', 'agency_std']

        df = df.merge(agency_stats, on='agency', how='left')

        # Calculate z-scores
        df['z_score'] = (df['awarded_amt'] - df['agency_mean']) / df['agency_std'].replace(0, 1)

        # Prepare features for anomaly detection
        anomaly_features = df[['awarded_amt', 'z_score']].fillna(0)

        # Train Isolation Forest
        self.anomaly_model = IsolationForest(
            contamination=contamination,
            random_state=42,
            n_estimators=100
        )

        df['anomaly_score'] = self.anomaly_model.fit_predict(anomaly_features)
        df['is_anomaly'] = df['anomaly_score'].apply(lambda x: 1 if x == -1 else 0)

        return df

    def generate_insights(self, df, monthly_data):
        """Generate actionable insights"""
        insights = {
            'total_spending': df['awarded_amt'].sum(),
            'avg_award': df['awarded_amt'].mean(),
            'total_transactions': len(df),
            'top_agencies': df.groupby('agency')['awarded_amt'].sum().nlargest(5).to_dict(),
            'top_suppliers': df.groupby('supplier_name')['awarded_amt'].sum().nlargest(5).to_dict(),
            'monthly_trend': monthly_data[['month', 'total_spending']].to_dict('records'),
            'forecast_accuracy': {},
            'anomaly_summary': {
                'count': len(df[df['is_anomaly'] == 1]),
                'percentage': len(df[df['is_anomaly'] == 1]) / len(df) * 100,
                'top_anomalies': df[df['is_anomaly'] == 1].nlargest(5, 'awarded_amt')[
                    ['agency', 'supplier_name', 'awarded_amt', 'award_date']
                ].to_dict('records')
            }
        }

        return insights

# Initialize session state
if 'analytics' not in st.session_state:
    st.session_state.analytics = ProcurementAnalytics()
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False
if 'insights' not in st.session_state:
    st.session_state.insights = None
if 'df' not in st.session_state:
    st.session_state.df = None
if 'monthly_data' not in st.session_state:
    st.session_state.monthly_data = None
if 'forecast_results' not in st.session_state:
    st.session_state.forecast_results = None
if 'yearly_forecast' not in st.session_state:
    st.session_state.yearly_forecast = None

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E3A8A;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #F8FAFC;
        border-radius: 10px;
        padding: 1.5rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .anomaly-card {
        background-color: #FEF2F2;
        border-left: 4px solid #DC2626;
    }
    .success-card {
        background-color: #F0FDF4;
        border-left: 4px solid #16A34A;
    }
    .forecast-card {
        background-color: #EFF6FF;
        border-left: 4px solid #3B82F6;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        border-radius: 4px 4px 0px 0px;
    }
</style>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.image("kendos.png", width=80)
    st.title("📊 Procurement Analytics")
    st.markdown("---")

    st.subheader("📁 Data Upload")
    uploaded_file = st.file_uploader(
        "Upload procurement data (CSV)",
        type=['csv'],
        help="Upload a CSV file with procurement data including award dates and amounts"
    )

    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            with st.spinner("Processing data..."):
                processed_df = st.session_state.analytics.load_and_preprocess(df)

                if processed_df is not None:
                    st.session_state.df = processed_df
                    st.session_state.df = st.session_state.analytics.create_features(st.session_state.df)
                    st.session_state.data_loaded = True
                    st.success(f"✅ Data loaded successfully! ({len(st.session_state.df):,} records)")
        except Exception as e:
            st.error(f"Error loading file: {str(e)}")

    st.markdown("---")

    if st.session_state.data_loaded:
        st.subheader("⚙️ Analysis Settings")

        col1, col2 = st.columns(2)
        with col1:
            anomaly_contamination = st.slider(
                "Anomaly Sensitivity",
                min_value=0.01,
                max_value=0.20,
                value=0.05,
                step=0.01,
                help="Higher values detect more anomalies"
            )

        with col2:
            forecast_months = st.number_input(
                "Forecast Months",
                min_value=1,
                max_value=36,
                value=12,
                step=1,
                help="Number of months to forecast"
            )

        if st.button("🚀 Run Full Analysis", type="primary", use_container_width=True):
            with st.spinner("Running analysis..."):
                # Prepare data
                monthly_data = st.session_state.analytics.prepare_forecast_data(st.session_state.df)
                st.session_state.monthly_data = monthly_data

                # Train forecast model
                mae, r2, predictions, features = st.session_state.analytics.train_forecast_model(monthly_data)

                # Generate forecasts
                st.session_state.forecast_results = st.session_state.analytics.predict_future_months(forecast_months)
                st.session_state.yearly_forecast = st.session_state.analytics.predict_yearly_summary(
                    n_years=max(1, forecast_months // 12)
                )

                # Detect anomalies
                df_with_anomalies = st.session_state.analytics.detect_anomalies(
                    st.session_state.df, 
                    contamination=anomaly_contamination
                )

                # Generate insights
                st.session_state.insights = st.session_state.analytics.generate_insights(df_with_anomalies, monthly_data)
                st.session_state.insights['forecast_accuracy'] = {
                    'mae': mae,
                    'r2': r2
                }
                st.session_state.df = df_with_anomalies

                st.success("Analysis complete!")

        st.markdown("---")
        st.subheader("💾 Export Results")

        if st.session_state.insights:
            col_exp1, col_exp2 = st.columns(2)

            with col_exp1:
                if st.button("📥 Anomalies Report", use_container_width=True):
                    anomalies_df = st.session_state.df[st.session_state.df['is_anomaly'] == 1]
                    csv = anomalies_df.to_csv(index=False)
                    st.download_button(
                        label="Download CSV",
                        data=csv,
                        file_name="procurement_anomalies.csv",
                        mime="text/csv",
                        key="anomalies_download"
                    )

            with col_exp2:
                if st.button("📈 Forecast Data", use_container_width=True) and st.session_state.forecast_results is not None:
                    csv = st.session_state.forecast_results.to_csv(index=False)
                    st.download_button(
                        label="Download CSV",
                        data=csv,
                        file_name="procurement_forecast.csv",
                        mime="text/csv",
                        key="forecast_download"
                    )

# Main content
st.markdown('<h1 class="main-header">🏛️ Government Procurement Analytics Dashboard</h1>', unsafe_allow_html=True)

if not st.session_state.data_loaded:
    # Welcome screen
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.info("👈 Please upload your procurement data using the sidebar to begin analysis.")

        st.markdown("""
        ### Expected Data Format:
        Your CSV file should include at minimum:
        - **award_date**: Date of award (YYYY-MM-DD or similar)
        - **awarded_amt**: Award amount (numeric)
        - **agency**: Government agency name
        - **supplier_name**: Supplier/vendor name

        ### Features Included:
        ✅ **Advanced Spending Forecasting** (Monthly & Yearly)  
        ✅ **Anomaly Detection**  
        ✅ **Agency Performance Analysis**  
        ✅ **Interactive Visualizations**  
        ✅ **Exportable Reports**  
        ✅ **Confidence Intervals**
        """)
else:
    # Display metrics
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Total Spending", f"${st.session_state.df['awarded_amt'].sum():,.0f}")
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Total Transactions", f"{len(st.session_state.df):,}")
        st.markdown('</div>', unsafe_allow_html=True)

    with col3:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        avg_award = st.session_state.df['awarded_amt'].mean()
        st.metric("Average Award", f"${avg_award:,.0f}")
        st.markdown('</div>', unsafe_allow_html=True)

    with col4:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        agency_count = st.session_state.df['agency'].nunique()
        st.metric("Unique Agencies", f"{agency_count}")
        st.markdown('</div>', unsafe_allow_html=True)

    if st.session_state.insights:
        st.markdown("---")

        # Create tabs for different sections
        tab1, tab2, tab3, tab4 = st.tabs([
            "📈 Spending Analysis", 
            "🔮 Spending Forecast", 
            "⚠️ Anomaly Detection",
            "📋 Data Explorer"
        ])

        with tab1:
            # Top row: Spending Trends
            col1, col2 = st.columns(2)

            with col1:
                st.subheader("📈 Monthly Spending Trend")
                monthly_data = st.session_state.monthly_data

                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=monthly_data['month'],
                    y=monthly_data['total_spending'],
                    mode='lines+markers',
                    name='Actual Spending',
                    line=dict(color='#3B82F6', width=2)
                ))
                fig.update_layout(
                    height=400,
                    xaxis_title="Month",
                    yaxis_title="Total Spending ($)",
                    template="plotly_white",
                    showlegend=True
                )
                st.plotly_chart(fig, use_container_width=True)

            with col2:
                st.subheader("🏢 Top Agencies by Spending")
                agency_spending = st.session_state.df.groupby('agency')['awarded_amt'].sum().nlargest(10)

                fig = go.Figure(data=[
                    go.Bar(
                        x=agency_spending.values,
                        y=agency_spending.index,
                        orientation='h',
                        marker_color='#10B981',
                        text=[f'${x:,.0f}' for x in agency_spending.values],
                        textposition='auto'
                    )
                ])
                fig.update_layout(
                    height=400,
                    xaxis_title="Total Spending ($)",
                    yaxis_title="Agency",
                    template="plotly_white"
                )
                st.plotly_chart(fig, use_container_width=True)

            # Second row: Distribution and Seasonality
            col1, col2 = st.columns(2)

            with col1:
                st.subheader("📊 Spending Distribution")

                fig = go.Figure(data=[
                    go.Histogram(
                        x=st.session_state.df['awarded_amt'],
                        nbinsx=50,
                        marker_color='#8B5CF6',
                        opacity=0.7
                    )
                ])
                fig.update_layout(
                    height=300,
                    xaxis_title="Award Amount ($)",
                    yaxis_title="Frequency",
                    template="plotly_white",
                    xaxis=dict(tickformat="$,.0f")
                )
                st.plotly_chart(fig, use_container_width=True)

            with col2:
                st.subheader("📅 Monthly Pattern")
                monthly_avg = st.session_state.df.groupby('month')['awarded_amt'].mean().reset_index()

                fig = go.Figure(data=[
                    go.Bar(
                        x=['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                           'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'],
                        y=[monthly_avg[monthly_avg['month'] == m]['awarded_amt'].mean() 
                           if m in monthly_avg['month'].values else 0 
                           for m in range(1, 13)],
                        marker_color='#F59E0B'
                    )
                ])
                fig.update_layout(
                    height=300,
                    xaxis_title="Month",
                    yaxis_title="Average Spending ($)",
                    template="plotly_white",
                    yaxis=dict(tickformat="$,.0f")
                )
                st.plotly_chart(fig, use_container_width=True)

        with tab2:
            if st.session_state.forecast_results is not None and st.session_state.yearly_forecast is not None:
                col1, col2 = st.columns(2)

                with col1:
                    st.subheader("📈 Monthly Forecast")

                    # Combine historical and forecast data
                    historical = st.session_state.monthly_data[['month', 'total_spending']].copy()
                    historical['type'] = 'Historical'

                    forecast = st.session_state.forecast_results[['month', 'predicted_spending']].copy()
                    forecast = forecast.rename(columns={'predicted_spending': 'total_spending'})
                    forecast['type'] = 'Forecast'

                    combined = pd.concat([historical, forecast])

                    fig = go.Figure()

                    # Add historical data
                    fig.add_trace(go.Scatter(
                        x=historical['month'],
                        y=historical['total_spending'],
                        mode='lines',
                        name='Historical',
                        line=dict(color='#3B82F6', width=2)
                    ))

                    # Add forecast with confidence interval
                    fig.add_trace(go.Scatter(
                        x=forecast['month'],
                        y=st.session_state.forecast_results['predicted_spending'],
                        mode='lines',
                        name='Forecast',
                        line=dict(color='#10B981', width=2, dash='dash')
                    ))

                    fig.add_trace(go.Scatter(
                        x=forecast['month'].tolist() + forecast['month'].tolist()[::-1],
                        y=st.session_state.forecast_results['confidence_interval_high'].tolist() + 
                          st.session_state.forecast_results['confidence_interval_low'].tolist()[::-1],
                        fill='toself',
                        fillcolor='rgba(16, 185, 129, 0.2)',
                        line=dict(color='rgba(255,255,255,0)'),
                        name='80% Confidence Interval'
                    ))

                    fig.update_layout(
                        height=400,
                        xaxis_title="Month",
                        yaxis_title="Total Spending ($)",
                        template="plotly_white",
                        showlegend=True,
                        hovermode='x unified'
                    )
                    st.plotly_chart(fig, use_container_width=True)

                    # Display forecast metrics
                    col_metrics1, col_metrics2 = st.columns(2)
                    with col_metrics1:
                        total_forecast = st.session_state.forecast_results['predicted_spending'].sum()
                        st.metric("Total Forecasted Spending", f"${total_forecast:,.0f}")

                    with col_metrics2:
                        avg_monthly_forecast = st.session_state.forecast_results['predicted_spending'].mean()
                        st.metric("Average Monthly Forecast", f"${avg_monthly_forecast:,.0f}")

                with col2:
                    st.subheader("📊 Yearly Forecast Summary")

                    fig = go.Figure(data=[
                        go.Bar(
                            x=st.session_state.yearly_forecast['year'].astype(str),
                            y=st.session_state.yearly_forecast['predicted_spending'],
                            marker_color=['#3B82F6', '#10B981', '#8B5CF6', '#F59E0B'],
                            text=[f'${x:,.0f}' for x in st.session_state.yearly_forecast['predicted_spending']],
                            textposition='auto'
                        )
                    ])

                    # Add confidence interval error bars
                    fig.add_trace(go.Scatter(
                        x=st.session_state.yearly_forecast['year'].astype(str),
                        y=st.session_state.yearly_forecast['confidence_interval_high'],
                        mode='markers',
                        marker=dict(color='red', size=8, symbol='triangle-up'),
                        name='Upper Bound'
                    ))

                    fig.add_trace(go.Scatter(
                        x=st.session_state.yearly_forecast['year'].astype(str),
                        y=st.session_state.yearly_forecast['confidence_interval_low'],
                        mode='markers',
                        marker=dict(color='orange', size=8, symbol='triangle-down'),
                        name='Lower Bound'
                    ))

                    fig.update_layout(
                        height=400,
                        xaxis_title="Year",
                        yaxis_title="Predicted Spending ($)",
                        template="plotly_white",
                        showlegend=True,
                        yaxis=dict(tickformat="$,.0f")
                    )
                    st.plotly_chart(fig, use_container_width=True)

                    # Display forecast details
                    st.subheader("📋 Detailed Forecast")

                    forecast_df = st.session_state.forecast_results.copy()
                    forecast_df['month'] = forecast_df['month'].dt.strftime('%b %Y')
                    forecast_df['predicted_spending'] = forecast_df['predicted_spending'].round(0)
                    forecast_df['confidence_low'] = forecast_df['confidence_interval_low'].round(0)
                    forecast_df['confidence_high'] = forecast_df['confidence_interval_high'].round(0)

                    st.dataframe(
                        forecast_df[['month', 'predicted_spending', 'confidence_low', 'confidence_high']].style.format({
                            'predicted_spending': '${:,.0f}',
                            'confidence_low': '${:,.0f}',
                            'confidence_high': '${:,.0f}'
                        }),
                        use_container_width=True,
                        height=300
                    )

                # Model performance metrics
                st.subheader("📊 Forecast Model Performance")
                col_perf1, col_perf2 = st.columns(2)

                with col_perf1:
                    mae = st.session_state.insights['forecast_accuracy']['mae']
                    st.markdown(f'<div class="forecast-card">', unsafe_allow_html=True)
                    st.metric("Mean Absolute Error (MAE)", f"${mae:,.0f}")
                    st.markdown('</div>', unsafe_allow_html=True)

                with col_perf2:
                    r2 = st.session_state.insights['forecast_accuracy']['r2']
                    st.markdown(f'<div class="forecast-card">', unsafe_allow_html=True)
                    st.metric("R² Score (Accuracy)", f"{r2:.3f}")
                    st.markdown('</div>', unsafe_allow_html=True)
            else:
                st.info("Run the analysis to generate spending forecasts.")

        with tab3:
            st.subheader("⚠️ Anomaly Detection Results")
            if 'is_anomaly' in st.session_state.df.columns:
                anomalies = st.session_state.df[st.session_state.df['is_anomaly'] == 1]
                anomaly_rate = len(anomalies) / len(st.session_state.df) * 100

                col_a1, col_a2, col_a3 = st.columns(3)
                with col_a1:
                    st.markdown(f'<div class="anomaly-card">', unsafe_allow_html=True)
                    st.metric("Anomalies Detected", f"{len(anomalies):,}")
                    st.markdown('</div>', unsafe_allow_html=True)

                with col_a2:
                    st.markdown(f'<div class="anomaly-card">', unsafe_allow_html=True)
                    st.metric("Anomaly Rate", f"{anomaly_rate:.1f}%")
                    st.markdown('</div>', unsafe_allow_html=True)

                with col_a3:
                    anomaly_value = anomalies['awarded_amt'].sum()
                    st.markdown(f'<div class="anomaly-card">', unsafe_allow_html=True)
                    st.metric("Anomaly Value", f"${anomaly_value:,.0f}")
                    st.markdown('</div>', unsafe_allow_html=True)

                if len(anomalies) > 0:
                    # Top anomalies chart
                    col1, col2 = st.columns(2)

                    with col1:
                        st.subheader("Top Anomalies by Amount")
                        top_anomalies = anomalies.nlargest(10, 'awarded_amt')

                        fig = go.Figure(data=[
                            go.Bar(
                                x=top_anomalies['agency'] + " - " + top_anomalies['supplier_name'].str.slice(0, 20),
                                y=top_anomalies['awarded_amt'],
                                marker_color='#DC2626',
                                text=[f'${x:,.0f}' for x in top_anomalies['awarded_amt']],
                                textposition='auto'
                            )
                        ])
                        fig.update_layout(
                            height=300,
                            xaxis_title="Agency - Supplier",
                            yaxis_title="Amount ($)",
                            template="plotly_white",
                            xaxis_tickangle=45
                        )
                        st.plotly_chart(fig, use_container_width=True)

                    with col2:
                        st.subheader("Anomalies by Agency")
                        agency_anomalies = anomalies.groupby('agency')['is_anomaly'].count().nlargest(10)

                        fig = go.Figure(data=[
                            go.Pie(
                                labels=agency_anomalies.index,
                                values=agency_anomalies.values,
                                hole=0.4
                            )
                        ])
                        fig.update_layout(
                            height=300,
                            showlegend=True
                        )
                        st.plotly_chart(fig, use_container_width=True)

                    # Anomalies table
                    st.subheader("📋 Detailed Anomalies List")
                    anomalies_display = anomalies[['agency', 'supplier_name', 'awarded_amt', 'award_date']].copy()
                    anomalies_display['award_date'] = anomalies_display['award_date'].dt.strftime('%Y-%m-%d')

                    st.dataframe(
                        anomalies_display.style.format({
                            'awarded_amt': '${:,.0f}'
                        }).applymap(
                            lambda x: 'background-color: #FEF2F2' if pd.notnull(x) else '',
                            subset=['awarded_amt']
                        ),
                        use_container_width=True,
                        height=300
                    )
                else:
                    st.info("No anomalies detected with current sensitivity settings.")
            else:
                st.info("Run the analysis to detect anomalies.")
        with tab4:
            st.subheader("📋 Data Explorer")

            explorer_col1, explorer_col2 = st.columns([2, 1])

            with explorer_col1:
                tab_data1, tab_data2, tab_data3 = st.tabs(["All Transactions", "Anomalies Only", "Statistics"])

                with tab_data1:
                    st.dataframe(
                        st.session_state.df.head(100).style.format({
                            'awarded_amt': '${:,.0f}',
                            'award_date': lambda x: x.strftime('%Y-%m-%d') if pd.notnull(x) else ''
                        }),
                        use_container_width=True,
                        height=400
                    )

                with tab_data2:
                    anomalies_df = st.session_state.df[st.session_state.df['is_anomaly'] == 1]
                    if len(anomalies_df) > 0:
                        st.dataframe(
                            anomalies_df.style.format({
                                'awarded_amt': '${:,.0f}',
                                'award_date': lambda x: x.strftime('%Y-%m-%d') if pd.notnull(x) else ''
                            }).applymap(
                                lambda x: 'background-color: #FEF2F2' if pd.notnull(x) else '',
                                subset=['awarded_amt']
                            ),
                            use_container_width=True,
                            height=400
                        )
                    else:
                        st.info("No anomalies detected with current settings.")

                with tab_data3:
                    st.write("**Summary Statistics:**")
                    stats_df = st.session_state.df['awarded_amt'].describe().reset_index()
                    stats_df.columns = ['Statistic', 'Value']
                    stats_df['Value'] = stats_df['Value'].apply(lambda x: f"${x:,.0f}" if isinstance(x, (int, float)) else str(x))
                    st.table(stats_df)

            with explorer_col2:
                st.subheader("🔍 Data Filters")

                # Agency filter
                agencies = st.session_state.df['agency'].unique()
                selected_agency = st.selectbox(
                    "Filter by Agency",
                    options=['All'] + list(agencies[:50])  # Limit to first 50 for performance
                )

                # Amount range filter
                min_amount = float(st.session_state.df['awarded_amt'].min())
                max_amount = float(st.session_state.df['awarded_amt'].max())

                amount_range = st.slider(
                    "Amount Range ($)",
                    min_value=min_amount,
                    max_value=max_amount,
                    value=(min_amount, max_amount),
                    step=max(1, (max_amount - min_amount) / 100)
                )

                # Date range filter
                min_date = st.session_state.df['award_date'].min().date()
                max_date = st.session_state.df['award_date'].max().date()

                date_range = st.date_input(
                    "Date Range",
                    value=(min_date, max_date),
                    min_value=min_date,
                    max_value=max_date
                )

                # Apply filters
                filtered_df = st.session_state.df.copy()

                if selected_agency != 'All':
                    filtered_df = filtered_df[filtered_df['agency'] == selected_agency]

                filtered_df = filtered_df[
                    (filtered_df['awarded_amt'] >= amount_range[0]) & 
                    (filtered_df['awarded_amt'] <= amount_range[1])
                ]

                if len(date_range) == 2:
                    filtered_df = filtered_df[
                        (filtered_df['award_date'].dt.date >= date_range[0]) & 
                        (filtered_df['award_date'].dt.date <= date_range[1])
                    ]

                st.metric("Filtered Records", len(filtered_df))
                st.metric("Filtered Total", f"${filtered_df['awarded_amt'].sum():,.0f}")

    else:
        # Data preview without analysis
        st.markdown("---")
        st.subheader("📋 Data Preview")
        st.dataframe(
            st.session_state.df.head(100).style.format({
                'awarded_amt': '${:,.0f}',
                'award_date': lambda x: x.strftime('%Y-%m-%d') if pd.notnull(x) else ''
            }),
            use_container_width=True
        )

        st.info("Click 'Run Full Analysis' in the sidebar to generate insights and visualizations.")

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #6B7280; padding: 1rem;'>"
    "Government Procurement Analytics Dashboard • Includes Advanced Forecasting"
    "</div>",
    unsafe_allow_html=True
)