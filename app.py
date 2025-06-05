import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="AI-Enhanced PMT Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .stAlert {
        padding: 1rem;
        border-radius: 0.5rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        text-align: center;
    }
    .trend-positive {
        color: #00cc00;
    }
    .trend-negative {
        color: #ff4444;
    }
</style>
""", unsafe_allow_html=True)

# Sidebar navigation
st.sidebar.title("🤖 AI-Enhanced PMT")
st.sidebar.markdown("---")
page = st.sidebar.radio(
    "Select Dashboard",
    ["Cost Overrun Prediction", "PMT Trend Analysis"]
)

# Helper functions for data generation
def generate_project_data():
    """Generate mock project data with cost and progress information"""
    np.random.seed(42)
    
    # Historical data for 6 months
    dates = pd.date_range(end=datetime.now(), periods=180, freq='D')
    
    # Base project parameters
    total_budget = 1000000  # $1M
    project_duration = 180  # days
    
    # Generate progress data with some variance
    planned_progress = np.linspace(0, 100, len(dates))
    actual_progress = planned_progress + np.cumsum(np.random.normal(-0.5, 2, len(dates)))
    actual_progress = np.clip(actual_progress, 0, 100)
    
    # Generate cost data (with potential overrun pattern)
    planned_cost = planned_progress / 100 * total_budget
    cost_variance = np.cumsum(np.random.normal(500, 1000, len(dates)))
    actual_cost = planned_cost + cost_variance
    
    # Create DataFrame
    df = pd.DataFrame({
        'date': dates,
        'planned_progress': planned_progress,
        'actual_progress': actual_progress,
        'planned_cost': planned_cost,
        'actual_cost': actual_cost,
        'budget_remaining': total_budget - actual_cost
    })
    
    return df, total_budget

def generate_pmt_data():
    """Generate mock PMT data with various metrics"""
    np.random.seed(42)
    
    # Generate data for the last 90 days
    dates = pd.date_range(end=datetime.now(), periods=90, freq='D')
    
    # Fill rates with decreasing trend
    base_fill_rate = 85
    trend = -0.15  # Decreasing trend
    noise = np.random.normal(0, 3, len(dates))
    fill_rates = base_fill_rate + trend * np.arange(len(dates)) + noise
    fill_rates = np.clip(fill_rates, 0, 100)
    
    # Other metrics
    productivity = 75 + np.random.normal(0, 5, len(dates))
    resource_utilization = 80 + np.random.normal(0, 4, len(dates))
    quality_score = 90 + np.random.normal(0, 2, len(dates))
    
    df = pd.DataFrame({
        'date': dates,
        'fill_rate': fill_rates,
        'productivity': productivity,
        'resource_utilization': resource_utilization,
        'quality_score': quality_score
    })
    
    return df

def predict_cost_overrun(df, total_budget):
    """Simple ML model to predict cost overrun"""
    # Prepare features
    X = df.index.values.reshape(-1, 1)
    y = df['actual_cost'].values
    
    # Train model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)
    
    # Predict future costs (next 30 days)
    future_days = 30
    future_X = np.arange(len(df), len(df) + future_days).reshape(-1, 1)
    predicted_costs = model.predict(future_X)
    
    # Calculate predicted final cost
    current_cost = df['actual_cost'].iloc[-1]
    predicted_final_cost = predicted_costs[-1]
    
    # Determine if overrun is likely
    overrun_threshold = total_budget * 1.05  # 5% buffer
    is_overrun_likely = predicted_final_cost > overrun_threshold
    overrun_amount = max(0, predicted_final_cost - total_budget)
    overrun_percentage = (overrun_amount / total_budget) * 100
    
    return {
        'is_overrun_likely': is_overrun_likely,
        'predicted_final_cost': predicted_final_cost,
        'overrun_amount': overrun_amount,
        'overrun_percentage': overrun_percentage,
        'confidence': 0.85  # Mock confidence score
    }

def detect_trends(df, metric_name):
    """Detect trends in PMT metrics using linear regression"""
    X = np.arange(len(df)).reshape(-1, 1)
    y = df[metric_name].values
    
    model = LinearRegression()
    model.fit(X, y)
    
    slope = model.coef_[0]
    
    # Determine trend significance
    if abs(slope) > 0.1:
        if slope > 0:
            return "increasing", slope
        else:
            return "decreasing", slope
    else:
        return "stable", slope

# Main app logic
if page == "Cost Overrun Prediction":
    st.title("🚨 AI-Enhanced Cost Overrun Prediction")
    st.markdown("Real-time project monitoring with predictive analytics")
    
    # Generate data
    project_data, total_budget = generate_project_data()
    
    # Predict cost overrun
    prediction = predict_cost_overrun(project_data, total_budget)
    
    # Display alert if overrun is likely
    if prediction['is_overrun_likely']:
        st.error(f"""
        ⚠️ **COST OVERRUN ALERT**
        
        Our AI model predicts a potential cost overrun:
        - Predicted final cost: ${prediction['predicted_final_cost']:,.2f}
        - Estimated overrun: ${prediction['overrun_amount']:,.2f} ({prediction['overrun_percentage']:.1f}%)
        - Confidence level: {prediction['confidence']*100:.0f}%
        
        **Recommended Actions:**
        - Review current spending patterns
        - Identify cost-saving opportunities
        - Consider scope adjustments
        """)
    else:
        st.success("✅ Project is currently within budget expectations")
    
    # Metrics row
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        current_cost = project_data['actual_cost'].iloc[-1]
        st.metric(
            "Current Spent",
            f"${current_cost:,.0f}",
            delta=f"${current_cost - project_data['planned_cost'].iloc[-1]:,.0f}"
        )
    
    with col2:
        current_progress = project_data['actual_progress'].iloc[-1]
        st.metric(
            "Project Progress",
            f"{current_progress:.1f}%",
            delta=f"{current_progress - project_data['planned_progress'].iloc[-1]:.1f}%"
        )
    
    with col3:
        st.metric(
            "Total Budget",
            f"${total_budget:,.0f}"
        )
    
    with col4:
        remaining = total_budget - current_cost
        st.metric(
            "Budget Remaining",
            f"${remaining:,.0f}",
            delta=f"{(remaining/total_budget)*100:.1f}%"
        )
    
    # Charts
    st.markdown("### 📊 Project Dashboard")
    
    tab1, tab2 = st.tabs(["Cost Analysis", "Progress Tracking"])
    
    with tab1:
        # Cost burn chart
        fig_cost = go.Figure()
        
        fig_cost.add_trace(go.Scatter(
            x=project_data['date'],
            y=project_data['planned_cost'],
            name='Planned Cost',
            line=dict(color='blue', dash='dash')
        ))
        
        fig_cost.add_trace(go.Scatter(
            x=project_data['date'],
            y=project_data['actual_cost'],
            name='Actual Cost',
            line=dict(color='red')
        ))
        
        # Add budget line
        fig_cost.add_hline(
            y=total_budget,
            line_dash="dash",
            line_color="orange",
            annotation_text="Total Budget"
        )
        
        fig_cost.update_layout(
            title="Cost Burn Rate Analysis",
            xaxis_title="Date",
            yaxis_title="Cost ($)",
            hovermode='x unified',
            height=400
        )
        
        st.plotly_chart(fig_cost, use_container_width=True)
    
    with tab2:
        # Progress chart
        fig_progress = go.Figure()
        
        fig_progress.add_trace(go.Scatter(
            x=project_data['date'],
            y=project_data['planned_progress'],
            name='Planned Progress',
            line=dict(color='green', dash='dash')
        ))
        
        fig_progress.add_trace(go.Scatter(
            x=project_data['date'],
            y=project_data['actual_progress'],
            name='Actual Progress',
            line=dict(color='purple')
        ))
        
        fig_progress.update_layout(
            title="Project Progress Tracking",
            xaxis_title="Date",
            yaxis_title="Progress (%)",
            hovermode='x unified',
            height=400
        )
        
        st.plotly_chart(fig_progress, use_container_width=True)

elif page == "PMT Trend Analysis":
    st.title("📈 PMT Trend Analysis Dashboard")
    st.markdown("AI-powered insights into your project management metrics")
    
    # Generate PMT data
    pmt_data = generate_pmt_data()
    
    # Detect trends
    fill_rate_trend, fill_rate_slope = detect_trends(pmt_data, 'fill_rate')
    
    # Display trend alert
    if fill_rate_trend == "decreasing" and abs(fill_rate_slope) > 0.1:
        st.warning(f"""
        📉 **TREND ALERT: Decreasing Fill Rate**
        
        AI analysis has detected a significant downward trend in fill rates:
        - Current fill rate: {pmt_data['fill_rate'].iloc[-1]:.1f}%
        - Average decline: {abs(fill_rate_slope):.2f}% per day
        - 30-day projection: {pmt_data['fill_rate'].iloc[-1] + 30*fill_rate_slope:.1f}%
        
        **Recommended Actions:**
        - Investigate recruitment bottlenecks
        - Review candidate pipeline
        - Enhance sourcing strategies
        """)
    
    # Metrics overview
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        current_fill = pmt_data['fill_rate'].iloc[-1]
        prev_fill = pmt_data['fill_rate'].iloc[-7]
        st.metric(
            "Current Fill Rate",
            f"{current_fill:.1f}%",
            delta=f"{current_fill - prev_fill:.1f}%"
        )
    
    with col2:
        current_prod = pmt_data['productivity'].iloc[-1]
        prev_prod = pmt_data['productivity'].iloc[-7]
        st.metric(
            "Productivity",
            f"{current_prod:.1f}%",
            delta=f"{current_prod - prev_prod:.1f}%"
        )
    
    with col3:
        current_util = pmt_data['resource_utilization'].iloc[-1]
        prev_util = pmt_data['resource_utilization'].iloc[-7]
        st.metric(
            "Resource Utilization",
            f"{current_util:.1f}%",
            delta=f"{current_util - prev_util:.1f}%"
        )
    
    with col4:
        current_qual = pmt_data['quality_score'].iloc[-1]
        prev_qual = pmt_data['quality_score'].iloc[-7]
        st.metric(
            "Quality Score",
            f"{current_qual:.1f}%",
            delta=f"{current_qual - prev_qual:.1f}%"
        )
    
    # Trend charts
    st.markdown("### 📊 PMT Metrics Dashboard")
    
    # Fill rate trend with AI highlight
    fig_fill = go.Figure()
    
    fig_fill.add_trace(go.Scatter(
        x=pmt_data['date'],
        y=pmt_data['fill_rate'],
        name='Fill Rate',
        line=dict(color='blue', width=2)
    ))
    
    # Add trend line
    X = np.arange(len(pmt_data))
    model = LinearRegression()
    model.fit(X.reshape(-1, 1), pmt_data['fill_rate'])
    trend_line = model.predict(X.reshape(-1, 1))
    
    fig_fill.add_trace(go.Scatter(
        x=pmt_data['date'],
        y=trend_line,
        name='Trend',
        line=dict(color='red', dash='dash')
    ))
    
    fig_fill.update_layout(
        title="Fill Rate Trend Analysis",
        xaxis_title="Date",
        yaxis_title="Fill Rate (%)",
        hovermode='x unified',
        height=400
    )
    
    st.plotly_chart(fig_fill, use_container_width=True)
    
    # Multi-metric comparison
    metrics_df = pmt_data[['date', 'productivity', 'resource_utilization', 'quality_score']]
    metrics_melted = metrics_df.melt(id_vars=['date'], var_name='Metric', value_name='Value')
    
    fig_multi = px.line(
        metrics_melted,
        x='date',
        y='Value',
        color='Metric',
        title='Multi-Metric Performance Trends'
    )
    
    fig_multi.update_layout(
        xaxis_title="Date",
        yaxis_title="Score (%)",
        hovermode='x unified',
        height=400
    )
    
    st.plotly_chart(fig_multi, use_container_width=True)
    
    # AI Insights panel
    with st.expander("🤖 AI-Generated Insights", expanded=True):
        st.markdown("""
        ### Key Findings:
        
        1. **Fill Rate Decline**: The fill rate has shown a consistent downward trend over the past 90 days
        2. **Productivity Stable**: Despite fill rate challenges, productivity remains consistent
        3. **Quality Maintained**: Quality scores show no significant degradation
        
        ### Predictive Analysis:
        - If current trends continue, fill rate may drop below 70% within 30 days
        - Resource utilization shows slight improvement potential
        - No immediate quality concerns detected
        """)

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #666;'>
        AI-Enhanced PMT Dashboard | Powered by Machine Learning
    </div>
    """,
    unsafe_allow_html=True
) 