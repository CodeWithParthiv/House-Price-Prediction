import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestRegressor
import joblib
import os

# Load the data
df = pd.read_csv('Housing.csv')

# Convert categorical variables to numerical
categorical_columns = ['mainroad', 'guestroom', 'basement', 'hotwaterheating', 
                      'airconditioning', 'prefarea', 'furnishingstatus']

label_encoders = {}
for column in categorical_columns:
    label_encoders[column] = LabelEncoder()
    df[column] = label_encoders[column].fit_transform(df[column])

# Save label encoders
joblib.dump(label_encoders, 'label_encoders.joblib', compress=3)

# Prepare features and target
X = df.drop('price', axis=1)
y = df['price']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Save the scaler
joblib.dump(scaler, 'scaler.joblib', compress=3)

# Train the model with reduced parameters
model = RandomForestRegressor(
    n_estimators=50,  # Reduced from 100
    max_depth=10,     # Added max depth
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1
)
model.fit(X_train_scaled, y_train)

# Save the model with compression
joblib.dump(model, 'house_price_model.joblib', compress=3)

# Print model performance
train_score = model.score(X_train_scaled, y_train)
test_score = model.score(X_test_scaled, y_test)
print(f"Training R² score: {train_score:.4f}")
print(f"Testing R² score: {test_score:.4f}")

# Print file sizes
for file in ['house_price_model.joblib', 'scaler.joblib', 'label_encoders.joblib']:
    size = os.path.getsize(file) / (1024 * 1024)  # Convert to MB
    print(f"{file} size: {size:.2f} MB") 