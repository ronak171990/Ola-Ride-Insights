# Step 2: Data Cleaning & Preprocessing

import pandas as pd
import numpy as np

# Load dataset
file_path = "OLA_DataSet.xlsx"
df = pd.read_excel(file_path, sheet_name="July")

# --- 1. Handle missing values ---
# Payment_Method: replace NaN with 'Not Applicable'
df['Payment_Method'] = df['Payment_Method'].fillna("Not Applicable")

# Ratings: keep NaN (they are natural for cancelled rides), no imputation here.

# --- 2. Standardize categories ---
df['Booking_Status'] = df['Booking_Status'].str.strip().str.title()

# --- 3. Derived Features ---
df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
df['Day_Of_Week'] = df['Date'].dt.day_name()
df['Hour_Of_Day'] = pd.to_datetime(df['Time'], errors='coerce').dt.hour

# Revenue per Km (only for valid rides: distance > 0)
df['Revenue_per_Km'] = np.where(df['Ride_Distance'] > 0,
                                df['Booking_Value'] / df['Ride_Distance'],
                                np.nan)

# --- 4. Outlier handling ---
# Cap Ride_Distance at 50 km (since max observed = 49, it's safe)
df['Ride_Distance'] = np.where(df['Ride_Distance'] > 50, 50, df['Ride_Distance'])

# Cap Booking_Value at 3000 INR (since max observed = 2999)
df['Booking_Value'] = np.where(df['Booking_Value'] > 3000, 3000, df['Booking_Value'])

# --- 5. Save cleaned dataset ---
df.to_csv("ola_cleaned.csv", index=False)

print("✅ Data cleaning done. Cleaned file saved as ola_cleaned.csv")
print("Shape after cleaning:", df.shape)
print("Columns:", df.columns.tolist())
