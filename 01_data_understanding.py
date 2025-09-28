# Step 1: Data Understanding & Exploration

import pandas as pd

# Load Excel file (replace path with your dataset location)
file_path = "OLA_DataSet.xlsx"
df = pd.read_excel(file_path, sheet_name="July")

# -------------------
# 1. Basic Info
# -------------------
print("Shape of dataset:", df.shape)
print("\nColumns:\n", df.columns.tolist())
print("\nData types:\n")
print(df.dtypes)

# -------------------
# 2. Preview Data
# -------------------
print("\n--- First 5 rows ---")
print(df.head())

# -------------------
# 3. Missing Values
# -------------------
print("\nMissing values per column:\n")
print(df.isna().sum())

# -------------------
# 4. Key Variables Analysis
# -------------------
# Booking Status distribution
print("\nBooking Status counts:\n")
print(df['Booking_Status'].value_counts(dropna=False))

# Payment Method distribution
print("\nPayment Method counts:\n")
print(df['Payment_Method'].value_counts(dropna=False))

# Vehicle Type distribution
print("\nVehicle Type counts:\n")
print(df['Vehicle_Type'].value_counts(dropna=False))

# -------------------
# 5. Numeric Columns Summary
# -------------------
numeric_cols = ['Ride_Distance', 'Booking_Value', 'Driver_Ratings', 'Customer_Rating']
print("\nNumeric summary:\n")
print(df[numeric_cols].describe())

# -------------------
# 6. Top Customers by Ride Count
# -------------------
print("\nTop 10 Customers by Ride Count:\n")
print(df['Customer_ID'].value_counts().head(10))

# -------------------
# 7. Cancellation Reasons
# -------------------
if 'Canceled_Rides_by_Customer' in df.columns:
    print("\nTop Customer Cancellation Reasons:\n")
    print(df['Canceled_Rides_by_Customer'].value_counts().head(10))

if 'Canceled_Rides_by_Driver' in df.columns:
    print("\nTop Driver Cancellation Reasons:\n")
    print(df['Canceled_Rides_by_Driver'].value_counts().head(10))
