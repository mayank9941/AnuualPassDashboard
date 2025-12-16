import streamlit as st
import pandas as pd
import requests
import plotly.express as px
from prophet import Prophet

# --- CONFIGURATION ---
API_URL = "https://rajmargyatra.nhai.gov.in/nhai/api/annualpass/v2.0/passReport"
PASS_PRICE = 3000
TARGET_DATE = "2026-03-31"

st.set_page_config(page_title="NHAI Future Core", layout="wide")

# --- CUSTOM CSS ---
st.markdown("""
<style>
    [data-testid="stMetricValue"] {
        font-size: 26px;
        font-weight: bold;
        color: #007bff;
    }
    div.stButton > button {
        width: 100%;
        background-image: linear-gradient(to right, #007bff, #0062cc);
        color: white;
        font-weight: bold;
        border-radius: 8px;
        padding: 0.6rem 1rem;
        border: none;
    }
    div.stButton > button:hover {
        background-image: linear-gradient(to right, #0062cc, #004a99);
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        color: white;
    }
</style>
""", unsafe_allow_html=True)

# --- COLOR PALETTES ---
ACTUAL_COLORS = ["#C9E9FF", "#A7D8FF", "#7CC2FF", "#55A9FF"]
REVENUE_COLORS = ["#D3F8E2", "#A9EFD3", "#7DE3C4", "#52D2AD"]
FORECAST_COLORS = ["#C9F2FF", "#A9E7FF", "#7DD9FF", "#4EC8FF"]

# --- HELPER: MOBILE CHART CONFIG ---
def make_chart_static(fig):
    """
    Disables zoom, pan, and drag modes for mobile optimization.
    """
    fig.update_layout(
        dragmode=False,  # Disables box/lasso selection
        xaxis=dict(fixedrange=True),  # Disables X-axis zoom/pan
        yaxis=dict(fixedrange=True),  # Disables Y-axis zoom/pan
        margin=dict(l=10, r=10, t=30, b=10) # Tighter margins for mobile
    )
    return fig

# Config to hide the floating toolbar
PLOT_CONFIG = {'displayModeBar': False, 'scrollZoom': False}

def fetch_data():
    try:
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(API_URL, headers=headers)
        data = response.json()
        
        if 'payload' in data:
            df = pd.DataFrame(data['payload'])
            df = df[['pass_start_date', 'active_count', 'pending_count']]
            df['Date'] = pd.to_datetime(df['pass_start_date']).dt.date
            df['Active Passes'] = pd.to_numeric(df['active_count'])
            df['Pending'] = pd.to_numeric(df['pending_count'])
            return df
        else:
            return pd.DataFrame()
    except:
        return pd.DataFrame()

def format_crores(value):
    return f"₹ {value / 10000000:.2f} Cr"

# --- HEADER ---
col_header, col_btn = st.columns([4, 1])
with col_header:
    st.title("FASTag Annual Pass")
with col_btn:
    st.write("")
    if st.button("🔄 Refresh Live Data"):
        st.rerun()

df_raw = fetch_data()

if not df_raw.empty:
    st.header("1. Current Performance (Actuals)")

    df_raw['Date'] = pd.to_datetime(df_raw['Date'])
    df_sorted = df_raw.sort_values(by='Date', ascending=False)

    total_active = df_raw['Active Passes'].sum()
    total_revenue = total_active * PASS_PRICE

    latest_active = df_sorted['Active Passes'].iloc[0]
    latest_date_str = df_sorted['Date'].iloc[0].strftime('%d %b')

    kpi1, kpi2, kpi3 = st.columns(3)
    kpi1.metric("Total Active Passes", f"{total_active:,}")
    kpi2.metric("Total Revenue Generated", format_crores(total_revenue))
    kpi3.metric(f"Latest Daily Sales ({latest_date_str})", f"{latest_active:,}")

    st.divider()

    # MONTHLY AGGREGATION
    df_raw['Month_Year'] = df_raw['Date'].dt.to_period('M')
    monthly_df = df_raw.groupby('Month_Year')['Active Passes'].sum().reset_index()
    monthly_df['Month Label'] = monthly_df['Month_Year'].dt.strftime('%b %Y')
    monthly_df['Revenue (Cr)'] = (monthly_df['Active Passes'] * PASS_PRICE) / 10000000
    monthly_df = monthly_df.sort_values('Month_Year')

    col1, col2 = st.columns(2)

    # Monthly Volume (Actual)
    with col1:
        st.subheader("Monthly Volume (Actual)")
        max_vol = monthly_df['Active Passes'].max()
        
        fig_vol = px.bar(
            monthly_df, x='Month Label', y='Active Passes', text_auto=True,
            color='Active Passes', color_continuous_scale=ACTUAL_COLORS
        )
        fig_vol.update_traces(
            textfont_size=16, textfont_color="white", 
            textposition="outside", cliponaxis=False
        )
        fig_vol.update_layout(
            coloraxis_showscale=False, 
            yaxis=dict(range=[0, max_vol * 1.3])
        )
        # Apply Mobile Fixes
        fig_vol = make_chart_static(fig_vol)
        st.plotly_chart(fig_vol, use_container_width=True, config=PLOT_CONFIG)

    # Monthly Revenue (Actual)
    with col2:
        st.subheader("Monthly Revenue (Actual)")
        max_rev = monthly_df['Revenue (Cr)'].max()
        
        fig_rev = px.bar(
            monthly_df, x='Month Label', y='Revenue (Cr)', 
            text=[f'{val:.2f} Cr' for val in monthly_df['Revenue (Cr)']], 
            color='Revenue (Cr)', color_continuous_scale=REVENUE_COLORS
        )
        fig_rev.update_traces(
            textfont_size=16, textfont_color="white", 
            textposition="outside", cliponaxis=False
        )
        fig_rev.update_layout(
            yaxis=dict(title="Revenue (Cr)", range=[0, max_rev * 1.3]), 
            coloraxis_showscale=False
        )
        # Apply Mobile Fixes
        fig_rev = make_chart_static(fig_rev)
        st.plotly_chart(fig_rev, use_container_width=True, config=PLOT_CONFIG)

    # Daily Breakdown (Actual)
    st.subheader("Daily Breakdown (Actuals)")
    df_sorted['Month_Name'] = df_sorted['Date'].dt.strftime('%B %Y')
    available_months = df_sorted['Month_Name'].unique().tolist()

    selected_month = st.selectbox("Select Month (History):", available_months, key="hist_month")

    if selected_month:
        day_wise_df = df_sorted[df_sorted['Month_Name'] == selected_month].sort_values('Date')
        max_daily = day_wise_df['Active Passes'].max()

        fig_daily = px.bar(
            day_wise_df, x='Date', y='Active Passes', text_auto=True,
            color="Active Passes", color_continuous_scale=ACTUAL_COLORS
        )
        fig_daily.update_traces(
            textfont_size=14, textfont_color="white", 
            textposition="outside", cliponaxis=False
        )
        fig_daily.update_layout(
            coloraxis_showscale=False, 
            yaxis=dict(range=[0, max_daily * 1.3])
        )
        fig_daily.update_xaxes(dtick="D1", tickformat="%d %b")

        # Apply Mobile Fixes
        fig_daily = make_chart_static(fig_daily)
        st.plotly_chart(fig_daily, use_container_width=True, config=PLOT_CONFIG)

        calc_df = day_wise_df.copy()
        current_month_str = pd.Timestamp.now().strftime('%B %Y')

        if selected_month == current_month_str and not calc_df.empty:
            calc_df = calc_df.iloc[:-1]

        avg_sales = calc_df['Active Passes'].mean() if not calc_df.empty else 0
        median_sales = calc_df['Active Passes'].median() if not calc_df.empty else 0

        col_a, col_b = st.columns(2)
        col_a.metric("Avg Daily Sales (Excl. Today)", f"{avg_sales:,.0f}")
        col_b.metric("Median Daily Sales", f"{median_sales:,.0f}")

    # Raw data
    with st.expander("See Raw Data (Actuals)", expanded=False):
        df_disp = df_sorted.copy()
        df_disp['Date'] = df_disp['Date'].dt.strftime('%d %b %Y')
        st.dataframe(df_disp[['Date', 'Active Passes', 'Pending']], use_container_width=True, hide_index=True)

    st.markdown("---")

    st.header(f"2. Strategic Forecast (Until March 2026)")

    with st.spinner("Calculating AI Projections..."):
        training_df = df_sorted.iloc[1:].copy()

        prophet_df = training_df[['Date', 'Active Passes']].rename(columns={'Date': 'ds', 'Active Passes': 'y'})
        prophet_df = prophet_df.sort_values('ds')

        m = Prophet(
            yearly_seasonality=False,
            weekly_seasonality=True,
            daily_seasonality=False,
            changepoint_prior_scale=0.2
        )
        m.fit(prophet_df)

        last_date = prophet_df['ds'].max()
        target_date = pd.to_datetime(TARGET_DATE)
        days_to_predict = (target_date - last_date).days

        if days_to_predict > 0:
            future = m.make_future_dataframe(periods=days_to_predict)
            forecast = m.predict(future)

            future_forecast = forecast[forecast['ds'] > last_date].copy()
            future_forecast['yhat'] = future_forecast['yhat'].clip(lower=0)
            future_forecast['Predicted Sales'] = future_forecast['yhat'].round().astype(int)
            future_forecast['Date'] = future_forecast['ds'].dt.date

            future_total_sales = future_forecast['Predicted Sales'].sum()
            future_total_revenue = future_total_sales * PASS_PRICE

            grand_total_sales = total_active + future_total_sales
            grand_total_revenue = total_revenue + future_total_revenue

            fkpi1, fkpi2, fkpi3 = st.columns(3)
            fkpi1.metric("Projected Grand Total Sales", f"{grand_total_sales:,.0f}", delta=f"+{future_total_sales:,}")
            fkpi2.metric("Projected Grand Total Revenue", format_crores(grand_total_revenue), delta=f"+{format_crores(future_total_revenue)}")
            fkpi3.metric("Forecast Horizon", "March 2026")

            # Monthly Forecast
            future_forecast['Month_Year'] = future_forecast['ds'].dt.to_period('M')
            f_monthly_df = future_forecast.groupby('Month_Year')['Predicted Sales'].sum().reset_index()
            f_monthly_df['Month Label'] = f_monthly_df['Month_Year'].dt.strftime('%b %Y')
            f_monthly_df['Revenue (Cr)'] = (f_monthly_df['Predicted Sales'] * PASS_PRICE) / 10000000
            f_monthly_df = f_monthly_df.sort_values('Month_Year')

            fc1, fc2 = st.columns(2)

            # Forecast Volume
            with fc1:
                st.subheader("Projected Monthly Volume")
                max_f_vol = f_monthly_df['Predicted Sales'].max()
                
                fig_f_vol = px.bar(
                    f_monthly_df, x='Month Label', y='Predicted Sales', text_auto=True,
                    color='Predicted Sales', color_continuous_scale=FORECAST_COLORS
                )
                fig_f_vol.update_traces(
                    textfont_size=16, textfont_color="white", 
                    textposition="outside", cliponaxis=False
                )
                fig_f_vol.update_layout(
                    coloraxis_showscale=False, 
                    yaxis=dict(range=[0, max_f_vol * 1.3])
                )
                # Apply Mobile Fixes
                fig_f_vol = make_chart_static(fig_f_vol)
                st.plotly_chart(fig_f_vol, use_container_width=True, config=PLOT_CONFIG)

            # Forecast Revenue
            with fc2:
                st.subheader("Projected Monthly Revenue")
                max_f_rev = f_monthly_df['Revenue (Cr)'].max()
                
                fig_f_rev = px.bar(
                    f_monthly_df, x='Month Label', y='Revenue (Cr)', 
                    text=[f'{val:.2f} Cr' for val in f_monthly_df['Revenue (Cr)']],  
                    color='Revenue (Cr)', color_continuous_scale=FORECAST_COLORS
                )
                fig_f_rev.update_traces(
                    textfont_size=16, textfont_color="white", 
                    textposition="outside", cliponaxis=False
                )
                fig_f_rev.update_layout(
                    yaxis=dict(title="Revenue (Cr)", range=[0, max_f_rev * 1.3]), 
                    coloraxis_showscale=False
                )
                # Apply Mobile Fixes
                fig_f_rev = make_chart_static(fig_f_rev)
                st.plotly_chart(fig_f_rev, use_container_width=True, config=PLOT_CONFIG)

            # Daily Forecast
            st.subheader("Future Daily Breakdown (Select Month)")
            future_forecast['Month_Name'] = future_forecast['ds'].dt.strftime('%B %Y')
            f_available_months = future_forecast['Month_Name'].unique().tolist()

            f_selected_month = st.selectbox("Select Future Month:", f_available_months, key="future_month")

            if f_selected_month:
                f_day_wise = future_forecast[future_forecast['Month_Name'] == f_selected_month]
                max_f_daily = f_day_wise['Predicted Sales'].max()

                fig_f_daily = px.bar(
                    f_day_wise, x='Date', y='Predicted Sales', text_auto=True,
                    color='Predicted Sales', color_continuous_scale=FORECAST_COLORS
                )
                fig_f_daily.update_traces(
                    textfont_size=14, textfont_color="white", 
                    textposition="outside", cliponaxis=False
                )
                fig_f_daily.update_layout(
                    coloraxis_showscale=False, 
                    yaxis=dict(range=[0, max_f_daily * 1.3])
                )
                fig_f_daily.update_xaxes(dtick="D1", tickformat="%d %b")

                # Apply Mobile Fixes
                fig_f_daily = make_chart_static(fig_f_daily)
                st.plotly_chart(fig_f_daily, use_container_width=True, config=PLOT_CONFIG)

                f_avg = f_day_wise['Predicted Sales'].mean()
                f_median = f_day_wise['Predicted Sales'].median()

                fc_a, fc_b = st.columns(2)
                fc_a.metric(f"Projected Avg Daily Sales ({f_selected_month})", f"{f_avg:,.0f}")
                fc_b.metric(f"Projected Median Sales ({f_selected_month})", f"{f_median:,.0f}")

            # Raw forecast table
            with st.expander("See Raw Forecast Data"):
                f_disp = future_forecast[['Date', 'Predicted Sales']].copy()
                f_disp['Date'] = pd.to_datetime(f_disp['Date']).dt.strftime('%d %b %Y')
                st.dataframe(f_disp, use_container_width=True, hide_index=True)

else:
    st.warning("Connecting to NHAI API...")
