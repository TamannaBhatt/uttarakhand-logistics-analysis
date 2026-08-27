import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("data/uttarakhand_logistics_cleaned.csv")
df["delay_days"] = df["actual_delivery_days"] - df["scheduled_delivery_days"]

print(df[[
    "distance_km", "shipment_volume_units",
    "warehouse_processing_days", "actual_delivery_days",
    "transportation_cost_inr"
]].describe())

plt.figure(figsize=(8, 5))
plt.hist(df["actual_delivery_days"], bins=20)
plt.xlabel("Actual Delivery Days")
plt.ylabel("Number of Shipments")
plt.title("Distribution of Actual Delivery Time")
plt.show()

plt.figure(figsize=(8, 5))
df.boxplot(column="actual_delivery_days", by="shipping_mode", grid=False)
plt.suptitle("")
plt.title("Delivery Time by Shipping Mode")
plt.show()

plt.figure(figsize=(8, 5))
plt.scatter(df["distance_km"], df["transportation_cost_inr"], alpha=0.45)
plt.xlabel("Distance (km)")
plt.ylabel("Transportation Cost (INR)")
plt.title("Transportation Cost vs Route Distance")
plt.show()

print(df[[
    "distance_km","shipment_volume_units","warehouse_processing_days",
    "weather_delay","traffic_delay","carrier_rating",
    "scheduled_delivery_days","actual_delivery_days",
    "transportation_cost_inr","late_delivery"
]].corr())
