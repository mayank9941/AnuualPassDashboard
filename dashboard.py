import streamlit as st
import pandas as pd
import requests
import plotly.express as px
from prophet import Prophet

# --- CONFIGURATION ---
API_URL = "https://rajmargyatra.nhai.gov.in/nhai/api/annualpass/v2.0/passReport"
PASS_PRICE_OLD = 3000
PASS_PRICE_NEW = 3075
PRICE_CHANGE_DATE = pd.Timestamp("2026-04-01")
TARGET_DATE = "2026-07-31"

st.set_page_config(page_title="NHAI Future Core", layout="wide")

# --- MOBILE-FIRST CSS ---
st.markdown("""
<style>
    /* Metric value sizing */
    [data-testid="stMetricValue"] {
        font-size: 22px;
        font-weight: bold;
        color: #007bff;
    }

    /* Refresh button */
    div.stButton > button {
        width: 100%;
        background-image: linear-gradient(to right, #007bff, #0062cc);
        color: white;
        font-weight: bold;
        border-radius: 8px;
        padding: 0.6rem 1rem;
        border: none;
        z-index: 99;
    }

    div[data-testid="stSelectbox"], div[data-testid="stExpander"] {
        position: relative;
        z-index: 100;
    }

    /* Stack 2-column chart rows on mobile */
    @media (max-width: 640px) {
        [data-testid="stMetricValue"] {
            font-size: 18px;
        }
        /* Make side-by-side columns stack vertically */
        div[data-testid="stHorizontalBlock"] {
            flex-wrap: wrap !important;
        }
        div[data-testid="stHorizontalBlock"] > div[data-testid="stVerticalBlock"] {
            min-width: 100% !important;
            flex: 1 1 100% !important;
        }
        /* Tighten metric delta text */
        [data-testid="stMetricDelta"] {
            font-size: 13px;
        }
    }

    /* Stat card style for avg/median table */
    .stat-card {
        background: #f0f8ff;
        border-radius: 10px;
        padding: 10px 14px;
        margin-bottom: 8px;
        display: flex;
        justify-content: space-between;
        align-items: center;
        font-size: 15px;
    }
    .stat-card .month { font-weight: 700; color: #0062cc; }
    .stat-card .vals { color: #333; }
</style>
""", unsafe_allow_html=True)

# --- COLOR PALETTES ---
ACTUAL_COLORS   = ["#C9E9FF", "#A7D8FF", "#7CC2FF", "#55A9FF"]
REVENUE_COLORS  = ["#D3F8E2", "#A9EFD3", "#7DE3C4", "#52D2AD"]
FORECAST_COLORS = ["#C9F2FF", "#A9E7FF", "#7DD9FF", "#4EC8FF"]

# --- HELPERS ---
def make_chart_static(fig):
    fig.update_layout(
        dragmode=False,
        xaxis=dict(fixedrange=True),
        yaxis=dict(fixedrange=True),
        margin=dict(l=10, r=10, t=30, b=10),
        hovermode="x unified"
    )
    return fig

PLOT_CONFIG = {'displayModeBar': False, 'scrollZoom': False, 'staticPlot': False}

def get_pass_price(date):
    return PASS_PRICE_NEW if pd.Timestamp(date) >= PRICE_CHANGE_DATE else PASS_PRICE_OLD

def format_crores(value):
    return f"₹ {value / 10000000:.2f} Cr"

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
        return pd.DataFrame()
    except:
        return pd.DataFrame()

# --- HEADER ---
col_header, col_btn = st.columns([4, 1])
with col_header:
    st.title("FASTag Annual Pass")
with col_btn:
    st.write("")
    if st.button("🔄 Refresh"):
        st.rerun()

df_raw = fetch_data()

if not df_raw.empty:
    # ── SECTION 1: ACTUALS ──────────────────────────────────────────────────
    st.header("1. Current Performance (Actuals)")

    df_raw['Date'] = pd.to_datetime(df_raw['Date'])
    df_sorted = df_raw.sort_values(by='Date', ascending=False)

    total_active  = df_raw['Active Passes'].sum()
    total_revenue = df_raw.apply(lambda r: r['Active Passes'] * get_pass_price(r['Date']), axis=1).sum()
    latest_active   = df_sorted['Active Passes'].iloc[0]
    latest_date_str = df_sorted['Date'].iloc[0].strftime('%d %b')

    # KPIs — 2 cols on mobile, 3 on desktop (CSS stacking handles it)
    kpi1, kpi2, kpi3 = st.columns(3)
    kpi1.metric("Total Active Passes", f"{total_active:,}")
    kpi2.metric("Total Revenue", format_crores(total_revenue))
    kpi3.metric("Latest Daily Sales", f"{latest_active:,}")

    st.divider()

    # Monthly actuals — stacked on mobile via CSS
    df_raw['Month_Year'] = df_raw['Date'].dt.to_period('M')
    monthly_df = df_raw.groupby('Month_Year').agg(
        active_passes=('Active Passes', 'sum'),
        first_date=('Date', 'first')
    ).reset_index()
    monthly_df.rename(columns={'active_passes': 'Active Passes'}, inplace=True)
    monthly_df['Month Label']   = monthly_df['Month_Year'].dt.strftime('%b %Y')
    monthly_df['Revenue (Cr)']  = monthly_df.apply(
        lambda r: (r['Active Passes'] * get_pass_price(r['first_date'])) / 10_000_000, axis=1
    )
    monthly_df = monthly_df.sort_values('Month_Year')

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Monthly Volume (Actual)")
        fig_vol = px.bar(monthly_df, x='Month Label', y='Active Passes', text_auto=True,
                         color='Active Passes', color_continuous_scale=ACTUAL_COLORS)
        fig_vol.update_traces(textfont_size=13, textfont_color="white", textposition="outside", cliponaxis=False)
        fig_vol.update_layout(coloraxis_showscale=False, yaxis=dict(range=[0, monthly_df['Active Passes'].max() * 1.3]))
        st.plotly_chart(make_chart_static(fig_vol), use_container_width=True, config=PLOT_CONFIG)

    with col2:
        st.subheader("Monthly Revenue (Actual)")
        fig_rev = px.bar(monthly_df, x='Month Label', y='Revenue (Cr)',
                         text=[f'{v:.2f} Cr' for v in monthly_df['Revenue (Cr)']],
                         color='Revenue (Cr)', color_continuous_scale=REVENUE_COLORS)
        fig_rev.update_traces(textfont_size=13, textfont_color="white", textposition="outside", cliponaxis=False)
        fig_rev.update_layout(coloraxis_showscale=False, yaxis=dict(range=[0, monthly_df['Revenue (Cr)'].max() * 1.3]))
        st.plotly_chart(make_chart_static(fig_rev), use_container_width=True, config=PLOT_CONFIG)

    # Daily breakdown
    st.markdown("---")
    st.subheader("Daily Breakdown (Actuals)")
    df_sorted['Month_Name'] = df_sorted['Date'].dt.strftime('%B %Y')
    selected_month = st.selectbox("Select Month (History):", df_sorted['Month_Name'].unique().tolist(), key="hist_month")

    if selected_month:
        day_wise_df = df_sorted[df_sorted['Month_Name'] == selected_month].sort_values('Date')
        fig_daily = px.bar(day_wise_df, x='Date', y='Active Passes', text_auto=True,
                           color='Active Passes', color_continuous_scale=ACTUAL_COLORS)
        fig_daily.update_traces(textfont_size=12, textfont_color="white", textposition="outside", cliponaxis=False)
        fig_daily.update_layout(coloraxis_showscale=False,
                                yaxis=dict(range=[0, day_wise_df['Active Passes'].max() * 1.3]))
        fig_daily.update_xaxes(dtick="D1", tickformat="%d %b")
        st.plotly_chart(make_chart_static(fig_daily), use_container_width=True, config=PLOT_CONFIG)

        calc_df = day_wise_df.copy()
        if selected_month == pd.Timestamp.now().strftime('%B %Y') and not calc_df.empty:
            calc_df = calc_df.iloc[:-1]
        avg_s    = calc_df['Active Passes'].mean()   if not calc_df.empty else 0
        median_s = calc_df['Active Passes'].median() if not calc_df.empty else 0

        ca, cb = st.columns(2)
        ca.metric("Avg Daily Sales (Excl. Today)", f"{avg_s:,.0f}")
        cb.metric("Median Daily Sales", f"{median_s:,.0f}")

    with st.expander("See Raw Data (Actuals)", expanded=False):
        df_disp = df_sorted.copy()
        df_disp['Date'] = df_disp['Date'].dt.strftime('%d %b %Y')
        st.dataframe(df_disp[['Date', 'Active Passes', 'Pending']], use_container_width=True, hide_index=True)

    st.markdown("---")

    # ── SECTION 2: FORECAST ─────────────────────────────────────────────────
    st.header("2. Strategic Forecast (Until July 2026)")
    st.caption("📌 Pass price: ₹3,000 up to 31 Mar 2026 | ₹3,075 from 1 Apr 2026 onwards")

    with st.spinner("Calculating AI Projections..."):
        training_df = df_sorted.iloc[1:].copy()
        prophet_df  = training_df[['Date', 'Active Passes']].rename(
            columns={'Date': 'ds', 'Active Passes': 'y'}
        ).sort_values('ds')

        m = Prophet(yearly_seasonality=False, weekly_seasonality=True,
                    daily_seasonality=False, changepoint_prior_scale=0.2)
        m.fit(prophet_df)

        last_date      = prophet_df['ds'].max()
        target_date    = pd.to_datetime(TARGET_DATE)
        days_to_predict = (target_date - last_date).days

        if days_to_predict > 0:
            future   = m.make_future_dataframe(periods=days_to_predict)
            forecast = m.predict(future)

            future_forecast = forecast[forecast['ds'] > last_date].copy()
            future_forecast['yhat']            = future_forecast['yhat'].clip(lower=0)
            future_forecast['Predicted Sales'] = future_forecast['yhat'].round().astype(int)
            future_forecast['ds']              = pd.to_datetime(future_forecast['ds'])
            future_forecast['Date']            = future_forecast['ds'].dt.date
            future_forecast['Month_Year']      = future_forecast['ds'].dt.to_period('M')
            future_forecast['Month_Name']      = future_forecast['ds'].dt.strftime('%B %Y')
            future_forecast['Pass Price']      = future_forecast['ds'].apply(get_pass_price)
            future_forecast['Revenue']         = future_forecast['Predicted Sales'] * future_forecast['Pass Price']

            future_total_sales   = future_forecast['Predicted Sales'].sum()
            future_total_revenue = future_forecast['Revenue'].sum()
            grand_total_sales    = total_active + future_total_sales
            grand_total_revenue  = total_revenue + future_total_revenue

            fk1, fk2, fk3 = st.columns(3)
            fk1.metric("Projected Grand Total Sales",    f"{grand_total_sales:,.0f}",  delta=f"+{future_total_sales:,}")
            fk2.metric("Projected Grand Total Revenue",  format_crores(grand_total_revenue), delta=f"+{format_crores(future_total_revenue)}")
            fk3.metric("Forecast Horizon", "July 2026")

            # Monthly forecast aggregation
            f_monthly_df = future_forecast.groupby('Month_Year').agg(
                predicted_sales=('Predicted Sales', 'sum'),
                revenue=('Revenue', 'sum'),
                pass_price=('Pass Price', 'first')
            ).reset_index()
            f_monthly_df.rename(columns={'predicted_sales': 'Predicted Sales'}, inplace=True)
            f_monthly_df['Month Label']  = f_monthly_df['Month_Year'].dt.strftime('%b %Y')
            f_monthly_df['Revenue (Cr)'] = f_monthly_df['revenue'] / 10_000_000
            f_monthly_df['Price Label']  = f_monthly_df['pass_price'].apply(lambda p: f"₹{p:,}/pass")
            f_monthly_df = f_monthly_df.sort_values('Month_Year')

            # Blend actual data into current month bar
            current_month_period = pd.Timestamp.now().to_period('M')
            if current_month_period in f_monthly_df['Month_Year'].values:
                actual_cur       = df_raw[df_raw['Date'].dt.to_period('M') == current_month_period]
                actual_sales_cur = actual_cur['Active Passes'].sum()
                actual_rev_cur   = (actual_cur['Active Passes'] * PASS_PRICE_OLD).sum()
                idx = f_monthly_df[f_monthly_df['Month_Year'] == current_month_period].index[0]
                f_monthly_df.loc[idx, 'Predicted Sales'] += actual_sales_cur
                f_monthly_df.loc[idx, 'Revenue (Cr)']   += actual_rev_cur / 10_000_000

            # Charts — stack on mobile via CSS
            fc1, fc2 = st.columns(2)
            with fc1:
                st.subheader("Projected Monthly Volume")
                fig_f_vol = px.bar(f_monthly_df, x='Month Label', y='Predicted Sales',
                                   text_auto=True, color='Predicted Sales',
                                   color_continuous_scale=FORECAST_COLORS,
                                   hover_data={'Price Label': True})
                fig_f_vol.update_traces(textfont_size=13, textfont_color="white",
                                        textposition="outside", cliponaxis=False)
                fig_f_vol.update_layout(coloraxis_showscale=False,
                                        yaxis=dict(range=[0, f_monthly_df['Predicted Sales'].max() * 1.3]))
                st.plotly_chart(make_chart_static(fig_f_vol), use_container_width=True, config=PLOT_CONFIG)

            with fc2:
                st.subheader("Projected Monthly Revenue")
                fig_f_rev = px.bar(f_monthly_df, x='Month Label', y='Revenue (Cr)',
                                   text=[f'{v:.2f} Cr' for v in f_monthly_df['Revenue (Cr)']],
                                   color='Revenue (Cr)', color_continuous_scale=FORECAST_COLORS,
                                   hover_data={'Price Label': True})
                fig_f_rev.update_traces(textfont_size=13, textfont_color="white",
                                        textposition="outside", cliponaxis=False)
                fig_f_rev.update_layout(coloraxis_showscale=False,
                                        yaxis=dict(title="Revenue (Cr)",
                                                   range=[0, f_monthly_df['Revenue (Cr)'].max() * 1.3]))
                st.plotly_chart(make_chart_static(fig_f_rev), use_container_width=True, config=PLOT_CONFIG)

            st.info("💡 **Price Change Applied:** Revenue for Apr–Jul 2026 is calculated at ₹3,075/pass. Months up to Mar 2026 use ₹3,000/pass.")

            # ── Daily Forecast breakdown ──
            st.markdown("---")
            st.subheader("Future Daily Breakdown (Select Month)")

            month_order = (
                future_forecast[['Month_Name', 'Month_Year']]
                .drop_duplicates()
                .sort_values('Month_Year')['Month_Name']
                .tolist()
            )
            f_selected_month = st.selectbox("Select Future Month:", month_order, key="future_month")

            if f_selected_month:
                f_day_wise      = future_forecast[future_forecast['Month_Name'] == f_selected_month].copy()
                effective_price = f_day_wise['Pass Price'].iloc[0]
                current_month_str = pd.Timestamp.now().strftime('%B %Y')

                if f_selected_month == current_month_str:
                    # Blend actual + forecast for current month
                    actual_this_month = df_raw[
                        df_raw['Date'].dt.strftime('%B %Y') == f_selected_month
                    ][['Date', 'Active Passes']].copy()
                    actual_this_month['Date']    = actual_this_month['Date'].dt.date
                    actual_this_month['Type']    = 'Actual'
                    actual_this_month['Revenue'] = actual_this_month['Active Passes'] * effective_price
                    actual_this_month = actual_this_month.rename(columns={'Active Passes': 'Sales'})

                    forecast_this_month = f_day_wise[['Date', 'Predicted Sales', 'Revenue']].copy()
                    forecast_this_month['Type'] = 'Forecast'
                    forecast_this_month = forecast_this_month.rename(columns={'Predicted Sales': 'Sales'})

                    combined = pd.concat([actual_this_month, forecast_this_month], ignore_index=True).sort_values('Date')
                    color_map = {'Actual': '#55A9FF', 'Forecast': '#4EC8FF'}

                    fig_f_daily = px.bar(combined, x='Date', y='Sales', text_auto=True,
                                         color='Type', color_discrete_map=color_map)
                    fig_f_daily.update_traces(textfont_size=11, textfont_color="white",
                                              textposition="outside", cliponaxis=False)
                    fig_f_daily.update_layout(
                        yaxis=dict(range=[0, combined['Sales'].max() * 1.3]),
                        legend=dict(title="", orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                    )
                    fig_f_daily.update_xaxes(dtick="D1", tickformat="%d %b")
                    st.plotly_chart(make_chart_static(fig_f_daily), use_container_width=True, config=PLOT_CONFIG)

                    actual_rev   = actual_this_month['Revenue'].sum()
                    forecast_rev = forecast_this_month['Revenue'].sum()
                    st.caption(f"🟦 Actual (days elapsed) + 🩵 Forecast (remaining) | ₹{effective_price:,}/pass")
                    fc_a, fc_b, fc_c = st.columns(3)
                    fc_a.metric("Actual Revenue (so far)",      format_crores(actual_rev))
                    fc_b.metric("Forecast Revenue (remaining)", format_crores(forecast_rev))
                    fc_c.metric("Projected Month Total",        format_crores(actual_rev + forecast_rev))

                    all_sales = pd.concat([
                        actual_this_month[['Sales']].rename(columns={'Sales': 'v'}),
                        forecast_this_month[['Sales']].rename(columns={'Sales': 'v'})
                    ])
                    sm_a, sm_b = st.columns(2)
                    sm_a.metric("Daily Avg (full month)", f"{all_sales['v'].mean():,.0f}")
                    sm_b.metric("Daily Median (full month)", f"{all_sales['v'].median():,.0f}")

                else:
                    fig_f_daily = px.bar(f_day_wise, x='Date', y='Predicted Sales', text_auto=True,
                                         color='Predicted Sales', color_continuous_scale=FORECAST_COLORS)
                    fig_f_daily.update_traces(textfont_size=12, textfont_color="white",
                                              textposition="outside", cliponaxis=False)
                    fig_f_daily.update_layout(coloraxis_showscale=False,
                                              yaxis=dict(range=[0, f_day_wise['Predicted Sales'].max() * 1.3]))
                    fig_f_daily.update_xaxes(dtick="D1", tickformat="%d %b")
                    st.plotly_chart(make_chart_static(fig_f_daily), use_container_width=True, config=PLOT_CONFIG)

                    fc_a, fc_b, fc_c = st.columns(3)
                    fc_a.metric(f"Projected Avg",    f"{f_day_wise['Predicted Sales'].mean():,.0f}")
                    fc_b.metric(f"Projected Median", f"{f_day_wise['Predicted Sales'].median():,.0f}")
                    fc_c.metric(f"Month Revenue (@ ₹{effective_price:,})", format_crores(f_day_wise['Revenue'].sum()))

            with st.expander("See Raw Forecast Data"):
                f_disp = future_forecast[['Date', 'Predicted Sales', 'Pass Price', 'Revenue']].copy()
                f_disp['Date']       = pd.to_datetime(f_disp['Date']).dt.strftime('%d %b %Y')
                f_disp['Revenue (₹)'] = f_disp['Revenue'].apply(lambda x: f"₹{x:,.0f}")
                f_disp['Pass Price'] = f_disp['Pass Price'].apply(lambda x: f"₹{x:,}")
                st.dataframe(f_disp[['Date', 'Predicted Sales', 'Pass Price', 'Revenue (₹)']],
                             use_container_width=True, hide_index=True)

else:
    st.warning("Connecting to NHAI API...")
