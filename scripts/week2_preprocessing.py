import pandas as pd
from sklearn.preprocessing import StandardScaler

RAW = "data/uttarakhand_logistics_raw_simulated.csv"
OUT = "data/uttarakhand_logistics_cleaned.csv"

df = pd.read_csv(RAW)

# 1. Remove duplicate records
df = df.drop_duplicates()

# 2. Handle missing numeric values with the median
for col in [
    "carrier_rating",
    "warehouse_processing_days",
    "transportation_cost_inr"
]:
    df[col] = df[col].fillna(df[col].median())

# Handle missing categorical value with the mode
df["road_condition"] = df["road_condition"].fillna(
    df["road_condition"].mode()[0]
)

# 3. Detect and cap numeric outliers using the IQR rule
numeric_cols = [
    "distance_km", "transportation_cost_inr",
    "shipment_volume_units", "inventory_level_units",
    "demand_units", "warehouse_processing_days",
    "carrier_rating"
]

for col in numeric_cols:
    q1 = df[col].quantile(0.25)
    q3 = df[col].quantile(0.75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    df[col] = df[col].clip(lower, upper)

# 4. Save cleaned data
df.to_csv(OUT, index=False)

# 5. Standardize numeric variables for ML demonstration
scale_cols = [
    "distance_km", "shipment_volume_units",
    "inventory_level_units", "demand_units",
    "warehouse_processing_days", "carrier_rating",
    "scheduled_delivery_days", "actual_delivery_days",
    "transportation_cost_inr"
]

scaler = StandardScaler()
df_scaled = df.copy()
df_scaled[scale_cols] = scaler.fit_transform(df_scaled[scale_cols])
df_scaled.to_csv("data/uttarakhand_logistics_scaled.csv", index=False)

print("Preprocessing completed.")
