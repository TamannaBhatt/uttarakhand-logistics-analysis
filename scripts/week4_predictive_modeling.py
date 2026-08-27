import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

df = pd.read_csv("data/uttarakhand_logistics_cleaned.csv")

features = [
    "origin_district","destination_district","origin_terrain",
    "destination_terrain","distance_km","shipping_mode",
    "shipment_volume_units","inventory_level_units","demand_units",
    "warehouse_processing_days","weather_delay","traffic_delay",
    "road_condition","carrier_rating","scheduled_delivery_days"
]
X, y = df[features], df["actual_delivery_days"]

categorical = [
    "origin_district","destination_district","origin_terrain",
    "destination_terrain","shipping_mode","road_condition"
]
numeric = [c for c in features if c not in categorical]

preprocessor = ColumnTransformer([
    ("num", StandardScaler(), numeric),
    ("cat", OneHotEncoder(handle_unknown="ignore"), categorical)
])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42
)

models = {
    "Linear Regression": LinearRegression(),
    "Random Forest": RandomForestRegressor(
        n_estimators=250, max_depth=12,
        min_samples_leaf=2, random_state=42
    )
}

for name, model in models.items():
    pipe = Pipeline([
        ("preprocessor", preprocessor),
        ("model", model)
    ])
    pipe.fit(X_train, y_train)
    pred = pipe.predict(X_test)
    print(name)
    print("MAE:", round(mean_absolute_error(y_test, pred), 3))
    print("RMSE:", round(np.sqrt(mean_squared_error(y_test, pred)), 3))
    print("R2:", round(r2_score(y_test, pred), 3))
