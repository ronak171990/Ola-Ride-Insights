import pandas as pd
from sqlalchemy import create_engine, text

# 1. Connect
engine = create_engine("postgresql+psycopg2://postgres:ronak1790@localhost:5432/postgres")

# 2. Load CSV (recreate table)
df = pd.read_csv("ola_cleaned.csv")
df.to_sql("july", engine, if_exists="replace", index=False)
print("✅ Table 'july' created in Postgres")

# 3. Rename columns to lowercase (one by one)
renames = {
    "Booking_Status": "booking_status",
    "Payment_Method": "payment_method",
    "Vehicle_Type": "vehicle_type",
    "Ride_Distance": "ride_distance",
    "Booking_Value": "booking_value",
    "Driver_Ratings": "driver_ratings",
    "Customer_Rating": "customer_rating",
    "Customer_ID": "customer_id",
    "Booking_ID": "booking_id",
    "Date": "date",
    "Time": "time"
}

with engine.begin() as conn:  # begin() auto-commits each statement
    for old, new in renames.items():
        try:
            conn.execute(text(f'ALTER TABLE july RENAME COLUMN "{old}" TO {new};'))
            print(f"Renamed {old} → {new}")
        except Exception as e:
            print(f"⚠️ Could not rename {old}: {e}")

print("✅ All renames attempted")

# 4. Test query
query = """
SELECT booking_status, COUNT(*) AS total
FROM july
GROUP BY booking_status;
"""
result = pd.read_sql(query, engine)
print(result)
