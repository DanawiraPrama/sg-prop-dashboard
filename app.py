import streamlit as st
import pandas as pd
import plotly.express as px
import pickle
import datetime
import os

# --- Page Configuration ---
st.set_page_config(page_title="HDB Insights Pro", page_icon="🏢", layout="wide")

# --- Load Data & ML Model ---
@st.cache_data
def load_data():
    try:
        df25 = pd.read_csv('hdb_resale_full_2025.csv')
        df26 = pd.read_csv('hdb_resale_full_2026.csv')
        df = pd.concat([df25, df26], ignore_index=True)
        df['price_per_sqm'] = df['resale_price'] / df['floor_area_sqm']
        return df.sort_values('month')
    except Exception as e:
        st.error(f"Missing data files. Please ensure the 2025 and 2026 CSV files are in the same folder.")
        return pd.DataFrame()

@st.cache_resource
def load_model():
    try:
        with open("hdb_model.pkl", "rb") as f:
            return pickle.load(f)
    except Exception as e:
        return None

df = load_data()
assets = load_model()

if df.empty:
    st.stop()

# --- Sidebar Navigation & Filters ---
st.sidebar.title("🏢 HDB Insights")
st.sidebar.markdown("Property Analysis Engine")
st.sidebar.markdown("---")

# Navigation
page = st.sidebar.radio("Navigation", ["📊 Market Insights", "🔮 Price Predictor"])
st.sidebar.markdown("---")

# ==========================================
# VIEW 1: MARKET INSIGHTS
# ==========================================
if page == "📊 Market Insights":
    
    # --- Filters ---
    st.sidebar.subheader("Filters")
    
    # 1. Date Range Filter
    all_months = sorted(df['month'].unique())
    start_month, end_month = st.sidebar.select_slider(
        "Date Range",
        options=all_months,
        value=(all_months[0], all_months[-1])
    )
    
    # 2. Town Filter
    all_towns = sorted(df['town'].unique())
    selected_towns = st.sidebar.multiselect("Town", options=all_towns, default=all_towns[:6])
    
    # 3. Flat Type Filter
    all_flats = sorted(df['flat_type'].unique())
    selected_flats = st.sidebar.multiselect("Flat Type", options=all_flats, default=all_flats)

    # Apply Filters
    mask = (
        (df['month'] >= start_month) & 
        (df['month'] <= end_month) & 
        (df['town'].isin(selected_towns)) & 
        (df['flat_type'].isin(selected_flats))
    )
    filtered_df = df[mask]

    # --- Main Dashboard ---
    st.title("Dashboard Overview")
    st.caption(f"Showing Real Data ({start_month} to {end_month})")

    if filtered_df.empty:
        st.warning("No transactions found for the selected filters.")
    else:
        # --- KPIs ---
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Transactions", f"{len(filtered_df):,}")
        with col2:
            median_price = filtered_df['resale_price'].median()
            st.metric("Median Resale Price", f"S$ {median_price:,.0f}")
        with col3:
            median_psm = filtered_df['price_per_sqm'].median()
            st.metric("Median Price / Sqm", f"S$ {median_psm:,.0f}")
        
        st.markdown("---")

        # --- Charts Row 1 ---
        chart_col1, chart_col2 = st.columns(2)
        
        with chart_col1:
            st.subheader("Price Trend by Month")
            trend_data = filtered_df.groupby('month')['resale_price'].median().reset_index()
            fig_trend = px.line(trend_data, x='month', y='resale_price', markers=True)
            fig_trend.update_layout(xaxis_title="", yaxis_title="Median Price (S$)", hovermode="x unified")
            st.plotly_chart(fig_trend, use_container_width=True)

        with chart_col2:
            st.subheader("Median Price Comparison by Town")
            town_data = filtered_df.groupby('town')['resale_price'].median().reset_index().sort_values('resale_price')
            fig_town = px.bar(town_data, x='resale_price', y='town', orientation='h', color_discrete_sequence=['#10b981'])
            fig_town.update_layout(xaxis_title="Median Price (S$)", yaxis_title="")
            st.plotly_chart(fig_town, use_container_width=True)

        # --- Charts Row 2 ---
        st.subheader("Price vs. Floor Area")
        # Sample data if it's too large to keep the browser running smoothly
        scatter_data = filtered_df.sample(min(2000, len(filtered_df)))
        fig_scatter = px.scatter(
            scatter_data, 
            x='floor_area_sqm', 
            y='resale_price', 
            color='flat_type', 
            hover_data=['town', 'street_name', 'month'],
            opacity=0.7
        )
        fig_scatter.update_layout(xaxis_title="Floor Area (sqm)", yaxis_title="Resale Price (S$)")
        st.plotly_chart(fig_scatter, use_container_width=True)

        # --- Table ---
        st.subheader("Recent Transactions")
        display_cols = ['month', 'town', 'flat_type', 'street_name', 'floor_area_sqm', 'resale_price']
        st.dataframe(
            filtered_df[display_cols].sort_values(by=['month', 'resale_price'], ascending=[False, False]).head(50),
            use_container_width=True,
            hide_index=True
        )

# ==========================================
# VIEW 2: AI PRICE PREDICTOR
# ==========================================
elif page == "🔮 Price Predictor":
    st.title("🔮 AI Property Valuation")
    st.write("Enter property specifications to get an instant estimated market value based on historical ML trends.")
    st.markdown("---")

    if not assets:
        st.error("⚠️ Model not found! Please run `python3 train.py` first to generate the `hdb_model.pkl` file.")
    else:
        model = assets["model"]
        encoder = assets["encoder"]
        
        town_options = sorted(list(encoder.categories_[0]))
        flat_type_options = list(encoder.categories_[1])

        with st.form("predict_form"):
            col1, col2 = st.columns(2)
            
            with col1:
                town = st.selectbox("Location (Town)", options=town_options)
                flat_type = st.selectbox("Flat Type", options=flat_type_options, index=2)
                floor_area = st.number_input("Floor Area (sqm)", min_value=30, max_value=250, value=90)
                
            with col2:
                storey = st.slider("Storey Level (Midpoint)", min_value=1, max_value=50, value=8)
                lease_left = st.slider("Remaining Lease (Years)", min_value=40, max_value=99, value=75)
                
                # Hidden variables for model input (Current year/month)
                current_month = datetime.datetime.now().month
                current_year = datetime.datetime.now().year
                
            submit = st.form_submit_button("Calculate Estimated Price", use_container_width=True)
            
        if submit:
            # Package the input data exactly as the model expects it
            input_df = pd.DataFrame([{
                'town': town,
                'flat_type': flat_type,
                'floor_area_sqm': floor_area,
                'mid_storey': storey,
                'lease_years': lease_left,
                'year': current_year,
                'month_num': current_month
            }])
            
            # Encode categorical text inputs to numbers
            input_df[['town', 'flat_type']] = encoder.transform(input_df[['town', 'flat_type']])
            
            # Generate Prediction
            predicted_price = model.predict(input_df)[0]
            
            st.success("### Estimated Market Value")
            st.markdown(f"<h1 style='text-align: center; color: #10b981;'>S$ {predicted_price:,.0f}</h1>", unsafe_allow_html=True)
            st.caption("*Disclaimer: This calculation is powered by a Random Forest Machine Learning model utilizing open public data. It is for informational reference only.*")