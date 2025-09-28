# 🚖 Ola Ride Insights Dashboard

An advanced analytics project that explores **ride patterns, revenue, cancellations, and customer/driver ratings** from Ola Rides data.  
The project integrates **SQL, Power BI, and Streamlit** to deliver both static and interactive dashboards.

---

## 📌 Project Overview
This project provides **end-to-end analytics** for Ola ride data:

- **Data Cleaning & Preprocessing**: Handled missing values, converted data types, and added derived features.
- **SQL Query Development**: Extracted key insights (trends, cancellations, top customers, ratings).
- **Power BI Dashboard**: Designed rich visualizations for ride performance and business KPIs.
- **Streamlit App**: Built an interactive dashboard with filters, charts, and advanced insights.
- **Business Storytelling**: Presented insights for decision-making (revenue optimization, cancellation patterns, customer behavior).

---

## 🛠️ Tech Stack
- **Python** → Pandas, NumPy, SQLAlchemy, Plotly, Streamlit  
- **SQL** → PostgreSQL for data storage and queries  
- **Visualization** → Power BI, Plotly  
- **Data** → Ola Rides dataset (`ola_cleaned.csv`)  

---

## 📊 Key Features

### 🔧 Data Pipeline
- Cleaned dataset saved as `ola_cleaned.csv`  
- Loaded into **Postgres** for SQL exploration  
- Streamlit connects directly to Postgres  

### 📈 Executive Dashboard
- Total bookings, revenue, success rate, avg booking value  
- Daily & hourly performance trends  
- Weekday vs weekend analysis  

### 🚗 Vehicle Analytics
- Revenue and efficiency by vehicle type  
- Ride distribution breakdown  
- Vehicle performance summary  

### 💰 Revenue Intelligence
- Revenue by payment method  
- Funnel analysis by time of day  
- Heatmap (day vs hour revenue distribution)  

### ❌ Cancellation Analysis
- Breakdown of customer & driver cancellations  
- Cancellations by vehicle type  
- Daily cancellation trend & lost revenue  

### ⭐ Ratings Insights
- Customer vs driver ratings  
- Rating distribution histograms  
- Ratings by vehicle type  

### 🔍 Advanced Analytics
- Correlation heatmap  
- Customer segmentation (top spenders, trip patterns)  
- Business insights (peak hours, top vehicle, weekend impact)

---

## 📂 Project Structure
Ola-Ride-Insights/
├── 01_data_understanding.py # Initial exploration
├── 02_data_cleaning.py # Cleaning & preprocessing
├── 03_run_sql.py # Load data to Postgres
├── 03_sql_queries.sql # SQL queries for insights
├── 04_powerbi.pbix # Power BI dashboard file
├── app.py # Streamlit dashboard
├── ola_cleaned.csv # Cleaned dataset
└── README.md # Project documentation


---

## 🚀 How to Run

### 1️⃣ Setup Environment
```bash
git clone https://github.com/your-username/Ola-Ride-Insights.git
cd Ola-Ride-Insights

# Create venv (optional)
python -m venv venv
source venv/bin/activate   # Mac/Linux
venv\Scripts\activate      # Windows

# Install dependencies
pip install -r requirements.txt
