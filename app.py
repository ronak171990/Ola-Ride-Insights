import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
import matplotlib.pyplot as plt
from sqlalchemy import create_engine, text
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# === CONFIGURATION ===
st.set_page_config(
    page_title="Ola Ride Analytics",
    page_icon="🚖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# === CUSTOM CSS ===
st.markdown("""
<style>
    .main > div {
        padding: 1rem 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        padding-left: 20px;
        padding-right: 20px;
    }
    .sidebar-metric {
        background: #f0f2f6;
        padding: 0.5rem;
        border-radius: 5px;
        margin: 0.25rem 0;
    }
</style>
""", unsafe_allow_html=True)

# === ENHANCED DATABASE CONNECTION WITH CONNECTION POOLING ===
@st.cache_resource
def get_database_connection():
    """Create database connection with error handling"""
    try:
        engine = create_engine(
            "postgresql+psycopg2://postgres:ronak1790@localhost:5432/postgres",
            pool_size=10,
            max_overflow=20,
            pool_pre_ping=True,
            pool_recycle=3600
        )
        return engine
    except Exception as e:
        st.error(f"Database connection failed: {str(e)}")
        return None

# === OPTIMIZED DATA LOADING ===
@st.cache_data(ttl=300)  # Cache for 5 minutes
def load_data():
    """Load and preprocess data with enhanced error handling"""
    engine = get_database_connection()
    if engine is None:
        return pd.DataFrame()
    
    try:
        # Simple query first, then process in Python for better compatibility
        query = """
        SELECT 
            *,
            CASE 
                WHEN booking_status = 'Success' THEN 'Completed'
                ELSE 'Cancelled'
            END as ride_outcome
        FROM july 
        ORDER BY date
        """
        
        with engine.connect() as conn:
            df = pd.read_sql(text(query), conn)
        
        # Enhanced data preprocessing
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df["booking_value"] = pd.to_numeric(df["booking_value"], errors="coerce")
        df["ride_distance"] = pd.to_numeric(df["ride_distance"], errors="coerce")
        
        # Create derived features using pandas after loading
        df["hour_of_day"] = df["date"].dt.hour
        df["day_of_week"] = df["date"].dt.dayofweek  # Monday=0, Sunday=6
        df["revenue_per_km"] = df["booking_value"] / df["ride_distance"].replace(0, np.nan)
        df["day_name"] = df["date"].dt.day_name()
        df["is_weekend"] = df["day_of_week"].isin([5, 6])  # Saturday=5, Sunday=6
        df["time_of_day"] = pd.cut(df["hour_of_day"], 
                                   bins=[0, 6, 12, 18, 24], 
                                   labels=['Night', 'Morning', 'Afternoon', 'Evening'],
                                   include_lowest=True)
        
        return df
    
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return pd.DataFrame()

# === UTILITY FUNCTIONS ===
def format_currency(value):
    """Format currency values"""
    if pd.isna(value):
        return "₹0"
    return f"₹{value:,.0f}"

def calculate_growth_rate(current, previous):
    """Calculate growth rate percentage"""
    if previous == 0:
        return 0
    return ((current - previous) / previous) * 100

def create_metric_card(title, value, delta=None, delta_color="normal"):
    """Create enhanced metric display"""
    delta_html = ""
    if delta is not None:
        color = "green" if delta >= 0 else "red"
        arrow = "↗" if delta >= 0 else "↘"
        delta_html = f'<span style="color: {color}; font-size: 0.8em;">{arrow} {delta:+.1f}%</span>'
    
    st.markdown(f"""
    <div class="metric-card">
        <h3 style="margin: 0; font-size: 1.2em;">{title}</h3>
        <h2 style="margin: 0.5rem 0; font-size: 2em;">{value}</h2>
        {delta_html}
    </div>
    """, unsafe_allow_html=True)

# === MAIN APP ===
def main():
    # Load data
    df = load_data()
    
    if df.empty:
        st.error("No data available. Please check your database connection.")
        return
    
    # === HEADER ===
    st.title("🚖 Ola Ride Analytics Dashboard")
    st.markdown("*Advanced insights into ride patterns, revenue, and performance metrics*")
    
    # === SIDEBAR FILTERS ===
    with st.sidebar:
        st.header("🔧 Control Panel")
        
        # Date Range Filter
        st.subheader("📅 Date Range")
        min_date = df["date"].min().date() if not df["date"].isna().all() else datetime.now().date()
        max_date = df["date"].max().date() if not df["date"].isna().all() else datetime.now().date()
        
        date_range = st.date_input(
            "Select period:",
            value=[min_date, max_date],
            min_value=min_date,
            max_value=max_date
        )
        
        # Advanced Filters
        st.subheader("🎯 Advanced Filters")
        
        # Vehicle Type Filter
        vehicle_types = ["All"] + sorted(df["vehicle_type"].dropna().unique().tolist())
        selected_vehicles = st.multiselect(
            "🚗 Vehicle Types:",
            options=vehicle_types,
            default=["All"]
        )
        
        # Payment Method Filter
        payment_methods = ["All"] + sorted(df["payment_method"].dropna().unique().tolist())
        selected_payments = st.multiselect(
            "💳 Payment Methods:",
            options=payment_methods,
            default=["All"]
        )
        
        # Booking Status Filter
        booking_statuses = ["All"] + sorted(df["booking_status"].dropna().unique().tolist())
        selected_status = st.multiselect(
            "📋 Booking Status:",
            options=booking_statuses,
            default=["All"]
        )
        
        # Revenue Range Filter
        if not df["booking_value"].isna().all():
            min_revenue, max_revenue = st.slider(
                "💰 Revenue Range:",
                min_value=float(df["booking_value"].min()),
                max_value=float(df["booking_value"].max()),
                value=(float(df["booking_value"].min()), float(df["booking_value"].max())),
                format="₹%.0f"
            )
        else:
            min_revenue = max_revenue = 0
    
    # === APPLY FILTERS ===
    df_filtered = df.copy()
    
    # Date filter
    if isinstance(date_range, list) and len(date_range) == 2:
        start_date = pd.to_datetime(date_range[0])
        end_date = pd.to_datetime(date_range[1]) + timedelta(days=1)
        df_filtered = df_filtered[
            (df_filtered["date"] >= start_date) & 
            (df_filtered["date"] < end_date)
        ]
    
    # Other filters
    if "All" not in selected_vehicles:
        df_filtered = df_filtered[df_filtered["vehicle_type"].isin(selected_vehicles)]
    
    if "All" not in selected_payments:
        df_filtered = df_filtered[df_filtered["payment_method"].isin(selected_payments)]
    
    if "All" not in selected_status:
        df_filtered = df_filtered[df_filtered["booking_status"].isin(selected_status)]
    
    # Revenue filter
    df_filtered = df_filtered[
        (df_filtered["booking_value"] >= min_revenue) & 
        (df_filtered["booking_value"] <= max_revenue)
    ]
    
    # Add download button
    st.sidebar.divider()
    st.sidebar.download_button(
        label="📥 Download Filtered Data",
        data=df_filtered.to_csv(index=False).encode("utf-8"),
        file_name=f"ola_rides_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv",
        help="Download the current filtered dataset as CSV"
    )
    
    # === MAIN CONTENT TABS ===
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📈 Executive Dashboard", 
        "🚗 Vehicle Analytics", 
        "💰 Revenue Intelligence", 
        "❌ Cancellation Analysis",
        "⭐ Rating Insights",
        "🔍 Advanced Analytics"
    ])
    
    # === TAB 1: EXECUTIVE DASHBOARD ===
    with tab1:
        if df_filtered.empty:
            st.warning("No data matches the selected filters.")
            return
            
        # Key Metrics Row
        col1, col2, col3, col4 = st.columns(4)
        
        total_bookings = len(df_filtered)
        completed_rides = len(df_filtered[df_filtered["booking_status"] == "Success"])
        cancelled_rides = total_bookings - completed_rides
        total_revenue = df_filtered["booking_value"].sum()
        
        with col1:
            create_metric_card("Total Bookings", f"{total_bookings:,}")
        with col2:
            success_rate = (completed_rides / total_bookings * 100) if total_bookings > 0 else 0
            create_metric_card("Success Rate", f"{success_rate:.1f}%")
        with col3:
            create_metric_card("Total Revenue", format_currency(total_revenue))
        with col4:
            avg_booking_value = total_revenue / total_bookings if total_bookings > 0 else 0
            create_metric_card("Avg. Booking Value", format_currency(avg_booking_value))
        
        st.divider()
        
        # Time Series Analysis
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("📊 Daily Performance Trends")
            
            daily_metrics = df_filtered.groupby(df_filtered["date"].dt.date).agg({
                'booking_value': ['sum', 'mean', 'count'],
                'ride_distance': 'mean'
            }).round(2)
            
            daily_metrics.columns = ['Revenue', 'Avg_Booking', 'Rides', 'Avg_Distance']
            daily_metrics = daily_metrics.reset_index()
            
            if not daily_metrics.empty:
                fig = make_subplots(
                    rows=2, cols=1,
                    subplot_titles=('Daily Revenue & Rides', 'Average Booking Value'),
                    specs=[[{"secondary_y": True}], [{"secondary_y": False}]]
                )
                
                # Revenue and rides
                fig.add_trace(
                    go.Bar(name="Revenue", x=daily_metrics["date"], y=daily_metrics["Revenue"],
                          marker_color='lightblue'), row=1, col=1
                )
                fig.add_trace(
                    go.Scatter(name="Rides", x=daily_metrics["date"], y=daily_metrics["Rides"],
                              mode='lines+markers', line=dict(color='red')), row=1, col=1, secondary_y=True
                )
                
                # Average booking value
                fig.add_trace(
                    go.Scatter(name="Avg Booking", x=daily_metrics["date"], y=daily_metrics["Avg_Booking"],
                              mode='lines+markers', line=dict(color='green')), row=2, col=1
                )
                
                fig.update_layout(height=500, showlegend=True)
                fig.update_yaxes(title_text="Revenue (₹)", row=1, col=1)
                fig.update_yaxes(title_text="Number of Rides", row=1, col=1, secondary_y=True)
                fig.update_yaxes(title_text="Average Booking Value (₹)", row=2, col=1)
                
                st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("🕒 Hourly Distribution")
            
            hourly_data = df_filtered.groupby("hour_of_day").size().reset_index(name="rides")
            
            if not hourly_data.empty:
                fig = px.bar(hourly_data, x="hour_of_day", y="rides",
                            title="Rides by Hour",
                            color="rides",
                            color_continuous_scale="viridis")
                fig.update_layout(height=300, showlegend=False)
                st.plotly_chart(fig, use_container_width=True)
            
            st.subheader("📅 Day of Week")
            
            dow_data = df_filtered.groupby("day_name").size().reindex([
                'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'
            ]).reset_index(name="rides")
            
            if not dow_data.empty:
                fig = px.pie(dow_data, values="rides", names="day_name",
                            title="Weekly Distribution")
                fig.update_layout(height=300)
                st.plotly_chart(fig, use_container_width=True)
    
    # === TAB 2: VEHICLE ANALYTICS ===
    with tab2:
        st.subheader("🚗 Vehicle Performance Analysis")
        
        if df_filtered.empty:
            st.warning("No data available for vehicle analysis.")
        else:
            # Vehicle metrics
            vehicle_metrics = df_filtered.groupby("vehicle_type").agg({
                "booking_value": ["sum", "mean", "count"],
                "ride_distance": ["mean", "sum"],
                "revenue_per_km": "mean"
            }).round(2)
            
            vehicle_metrics.columns = ["Total_Revenue", "Avg_Revenue", "Rides", "Avg_Distance", "Total_Distance", "Revenue_per_KM"]
            vehicle_metrics = vehicle_metrics.reset_index()
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Revenue by vehicle type
                fig = px.bar(vehicle_metrics, x="vehicle_type", y="Total_Revenue",
                            title="Total Revenue by Vehicle Type",
                            color="Total_Revenue",
                            color_continuous_scale="plasma")
                st.plotly_chart(fig, use_container_width=True)
                
                # Efficiency metrics
                fig = px.scatter(vehicle_metrics, x="Avg_Distance", y="Revenue_per_KM",
                               size="Rides", color="vehicle_type",
                               title="Distance vs Revenue Efficiency",
                               hover_data=["Total_Revenue"])
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Rides distribution
                fig = px.pie(vehicle_metrics, values="Rides", names="vehicle_type",
                           title="Ride Distribution by Vehicle Type")
                st.plotly_chart(fig, use_container_width=True)
                
                # Performance table
                st.subheader("📋 Vehicle Performance Summary")
                vehicle_display = vehicle_metrics.copy()
                vehicle_display["Total_Revenue"] = vehicle_display["Total_Revenue"].apply(format_currency)
                vehicle_display["Avg_Revenue"] = vehicle_display["Avg_Revenue"].apply(format_currency)
                st.dataframe(vehicle_display, use_container_width=True)
    
    # === TAB 3: REVENUE INTELLIGENCE ===
    with tab3:
        st.subheader("💰 Revenue Deep Dive")
        
        if df_filtered.empty:
            st.warning("No revenue data available.")
        else:
            # Payment method analysis
            col1, col2 = st.columns(2)
            
            with col1:
                payment_revenue = df_filtered.groupby("payment_method")["booking_value"].agg(["sum", "mean", "count"]).round(2)
                payment_revenue.columns = ["Total", "Average", "Count"]
                payment_revenue = payment_revenue.reset_index()
                
                fig = px.treemap(payment_revenue, path=['payment_method'], values='Total',
                               title="Revenue Distribution by Payment Method")
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Revenue by time of day
                time_revenue = df_filtered.groupby("time_of_day")["booking_value"].sum().reset_index()
                
                fig = px.funnel(time_revenue, x="booking_value", y="time_of_day",
                              title="Revenue by Time of Day")
                st.plotly_chart(fig, use_container_width=True)
            
            # Revenue heatmap by day and hour
            st.subheader("🔥 Revenue Heatmap")
            
            heatmap_data = df_filtered.pivot_table(
                values="booking_value", 
                index="day_name", 
                columns="hour_of_day", 
                aggfunc="sum"
            ).fillna(0)
            
            # Reorder days
            day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
            heatmap_data = heatmap_data.reindex(day_order)
            
            fig = px.imshow(heatmap_data,
                          title="Revenue Heatmap (Day vs Hour)",
                          color_continuous_scale="RdYlBu_r",
                          aspect="auto")
            st.plotly_chart(fig, use_container_width=True)
    
    # === TAB 4: CANCELLATION ANALYSIS ===
    with tab4:
        st.subheader("❌ Cancellation Pattern Analysis")
        
        cancelled_rides = df_filtered[df_filtered["booking_status"] != "Success"]
        
        if cancelled_rides.empty:
            st.success("🎉 No cancellations in the selected period!")
        else:
            col1, col2 = st.columns(2)
            
            with col1:
                # Cancellation reasons
                cancellation_counts = cancelled_rides["booking_status"].value_counts()
                
                fig = px.pie(values=cancellation_counts.values, 
                           names=cancellation_counts.index,
                           title="Cancellation Reasons")
                st.plotly_chart(fig, use_container_width=True)
                
                # Cancellation by vehicle type
                cancel_vehicle = cancelled_rides.groupby("vehicle_type").size().reset_index(name="cancellations")
                
                fig = px.bar(cancel_vehicle, x="vehicle_type", y="cancellations",
                           title="Cancellations by Vehicle Type",
                           color="cancellations")
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Cancellation trends
                daily_cancellations = cancelled_rides.groupby(
                    cancelled_rides["date"].dt.date
                ).size().reset_index(name="cancellations")
                daily_cancellations.columns = ["date", "cancellations"]
                
                fig = px.line(daily_cancellations, x="date", y="cancellations",
                            title="Daily Cancellation Trends")
                st.plotly_chart(fig, use_container_width=True)
                
                # Lost revenue analysis
                lost_revenue = cancelled_rides.groupby("booking_status")["booking_value"].sum().reset_index()
                
                fig = px.bar(lost_revenue, x="booking_status", y="booking_value",
                           title="Lost Revenue by Cancellation Type",
                           color="booking_value")
                st.plotly_chart(fig, use_container_width=True)
    
    # === TAB 5: RATING INSIGHTS ===
    with tab5:
        st.subheader("⭐ Customer & Driver Rating Analysis")
        
        rating_columns = [col for col in ["driver_ratings", "customer_rating"] if col in df_filtered.columns]
        
        if not rating_columns:
            st.warning("No rating data available in the dataset.")
        else:
            col1, col2 = st.columns(2)
            
            for i, rating_col in enumerate(rating_columns):
                with col1 if i == 0 else col2:
                    rating_data = df_filtered[rating_col].dropna()
                    
                    if not rating_data.empty:
                        st.subheader(f"📊 {rating_col.replace('_', ' ').title()} Distribution")
                        
                        # Rating histogram
                        fig = px.histogram(rating_data, nbins=20,
                                         title=f"{rating_col.replace('_', ' ').title()} Distribution")
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Rating stats
                        st.metric(f"Average {rating_col.replace('_', ' ').title()}", f"{rating_data.mean():.2f} ⭐")
                        st.metric(f"Median {rating_col.replace('_', ' ').title()}", f"{rating_data.median():.2f} ⭐")
            
            # Rating by vehicle type
            if rating_columns:
                st.subheader("🚗 Ratings by Vehicle Type")
                
                rating_by_vehicle = df_filtered.groupby("vehicle_type")[rating_columns].mean().reset_index()
                
                fig = px.bar(rating_by_vehicle, x="vehicle_type", y=rating_columns,
                           title="Average Ratings by Vehicle Type",
                           barmode="group")
                st.plotly_chart(fig, use_container_width=True)
    
    # === TAB 6: ADVANCED ANALYTICS ===
    with tab6:
        st.subheader("🔬 Advanced Analytics & Insights")
        
        if df_filtered.empty:
            st.warning("No data available for advanced analysis.")
        else:
            # Correlation analysis
            st.subheader("📈 Feature Correlation Analysis")
            
            numeric_cols = df_filtered.select_dtypes(include=[np.number]).columns
            correlation_matrix = df_filtered[numeric_cols].corr()
            
            fig = px.imshow(correlation_matrix,
                          title="Feature Correlation Heatmap",
                          color_continuous_scale="RdBu")
            st.plotly_chart(fig, use_container_width=True)
            
            # Customer behavior analysis
            if "customer_id" in df_filtered.columns:
                st.subheader("👥 Customer Behavior Insights")
                
                customer_stats = df_filtered.groupby("customer_id").agg({
                    "booking_value": ["sum", "mean", "count"],
                    "ride_distance": "mean"
                }).round(2)
                
                customer_stats.columns = ["Total_Spent", "Avg_Booking", "Trip_Count", "Avg_Distance"]
                customer_stats = customer_stats.reset_index()
                
                # Top customers
                top_customers = customer_stats.nlargest(10, "Total_Spent")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    fig = px.bar(top_customers, x="customer_id", y="Total_Spent",
                               title="Top 10 Customers by Revenue")
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    fig = px.scatter(customer_stats, x="Trip_Count", y="Total_Spent",
                                   size="Avg_Distance", color="Avg_Booking",
                                   title="Customer Segmentation Analysis")
                    st.plotly_chart(fig, use_container_width=True)
            
            # Operational efficiency
            st.subheader("⚡ Operational Efficiency Metrics")
            
            efficiency_metrics = {
                "Average Revenue per Ride": df_filtered["booking_value"].mean(),
                "Average Distance per Ride": df_filtered["ride_distance"].mean(),
                "Revenue per Kilometer": df_filtered["revenue_per_km"].mean(),
                "Success Rate": len(df_filtered[df_filtered["booking_status"] == "Success"]) / len(df_filtered) * 100,
                "Weekend vs Weekday Revenue": df_filtered[df_filtered["is_weekend"]]["booking_value"].sum() / df_filtered[~df_filtered["is_weekend"]]["booking_value"].sum()
            }
            
            metric_df = pd.DataFrame(list(efficiency_metrics.items()), columns=["Metric", "Value"])
            
            col1, col2 = st.columns([1, 1])
            
            with col1:
                st.dataframe(metric_df, use_container_width=True)
            
            with col2:
                # Business insights
                st.subheader("💡 Key Business Insights")
                
                insights = []
                
                if df_filtered["is_weekend"].any():
                    weekend_avg = df_filtered[df_filtered["is_weekend"]]["booking_value"].mean()
                    weekday_avg = df_filtered[~df_filtered["is_weekend"]]["booking_value"].mean()
                    if weekend_avg > weekday_avg:
                        insights.append(f"🔸 Weekend rides generate {((weekend_avg/weekday_avg-1)*100):.1f}% more revenue on average")
                    else:
                        insights.append(f"🔸 Weekday rides generate {((weekday_avg/weekend_avg-1)*100):.1f}% more revenue on average")
                
                peak_hour = df_filtered.groupby("hour_of_day").size().idxmax()
                insights.append(f"🔸 Peak booking hour is {peak_hour}:00")
                
                if "vehicle_type" in df_filtered.columns:
                    top_vehicle = df_filtered["vehicle_type"].value_counts().index[0]
                    insights.append(f"🔸 Most popular vehicle type: {top_vehicle}")
                
                most_revenue_day = df_filtered.groupby("day_name")["booking_value"].sum().idxmax()
                insights.append(f"🔸 Highest revenue day: {most_revenue_day}")
                
                for insight in insights:
                    st.markdown(insight)

if __name__ == "__main__":
    main()