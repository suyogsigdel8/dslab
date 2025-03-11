# PROJECT ON TAXI FARE PREDICTION

import numpy as np
import pandas as pd

# Load dataset
df = pd.read_csv("dataset_for_taxi_fare_prediction.csv")

# Convert key to datetime and extract useful time features
df['key'] = pd.to_datetime(df['key'])
df['hour'] = df['key'].dt.hour
df['day_of_week'] = df['key'].dt.dayofweek

# Remove missing values
df = df.dropna()

# Define function to calculate distance between coordinates using Haversine formula
def haversine(lat1, lon1, lat2, lon2):
    R = 6371  # Radius of Earth in km
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    return R * c

# Compute distance_km feature
df['distance_km'] = df.apply(lambda row: haversine(row['pickup_latitude'], row['pickup_longitude'],
                                                    row['dropoff_latitude'], row['dropoff_longitude']), axis=1)

# Select features and target variable
X = df[['distance_km', 'hour', 'day_of_week', 'passenger_count']]
y = df['fare_amount']

# Add bias term (column of 1s) for the intercept
X.insert(0, 'bias', 1)

# Convert to NumPy arrays
X = X.to_numpy()
y = y.to_numpy().reshape(-1, 1)

# Compute weights using the Normal Equation: theta = (X^T * X)^(-1) * X^T * y
theta = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(y)

# Function to make predictions
def predict_fare(pickup_lat, pickup_lon, dropoff_lat, dropoff_lon, hour, day_of_week, passenger_count):
    distance = haversine(pickup_lat, pickup_lon, dropoff_lat, dropoff_lon)
    X_new = np.array([1, distance, hour, day_of_week, passenger_count]).reshape(1, -1)
    return X_new.dot(theta)[0, 0]

# Function to validate input coordinates
def is_valid_location(lat, lon):
    return 40 <= lat <= 41 and -74.5 <= lon <= -73.5

# User input for pickup and dropoff coordinates
print("Enter pickup and dropoff coordinates within NYC range (-74.5 to -73.5 longitude, 40 to 41 latitude):")

pickup_lat = float(input("Pickup Latitude: "))
pickup_lon = float(input("Pickup Longitude: "))
dropoff_lat = float(input("Dropoff Latitude: "))
dropoff_lon = float(input("Dropoff Longitude: "))

# Check if input coordinates are within valid range
if not (is_valid_location(pickup_lat, pickup_lon) and is_valid_location(dropoff_lat, dropoff_lon)):
    print("Error: Pickup or dropoff location is out of NYC range. Please enter valid coordinates.")
    exit()

hour = int(input("Enter hour of the day (0-23): "))
day_of_week = int(input("Enter day of the week (0=Monday, 6=Sunday): "))
passenger_count = int(input("Enter number of passengers: "))

# Predict fare
predicted_fare_usd = predict_fare(pickup_lat, pickup_lon, dropoff_lat, dropoff_lon, hour, day_of_week, passenger_count)

# Convert fare to NPR
exchange_rate_usd_to_npr = 139.675  # Update this value as needed
predicted_fare_npr = predicted_fare_usd * exchange_rate_usd_to_npr

print(f"Predicted Fare: ${predicted_fare_usd:.2f} USD")
print(f"Predicted Fare: {predicted_fare_npr:.2f} NPR")

# Evaluate model performance
y_pred = X.dot(theta)
mae = np.mean(np.abs(y - y_pred))
mse = np.mean((y - y_pred) ** 2)
r2 = 1 - (np.sum((y - y_pred) ** 2) / np.sum((y - np.mean(y)) ** 2))

print("Model Evaluation:")
print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"R² Score: {r2:.4f}")
