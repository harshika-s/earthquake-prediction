import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.metrics import mean_squared_error, accuracy_score, confusion_matrix
import pickle
from xgboost import XGBRegressor, XGBClassifier

# Load the dataset
data = pd.read_csv('dataset.csv')

# Features (Latitude, Longitude, Depth) and target (Magnitude)
X = data[['Latitude', 'Longitude', 'Depth']]
y_magnitude = data['Magnitude']  # Target column for regression

# Categorize Magnitude into Intensity Levels for Classification
def categorize_intensity(magnitude):
    """
    Categorize magnitude into intensity levels based on updated thresholds.
    """
    if magnitude < 4.5:  # Adjusted threshold for "Low"
        return "Low"
    elif 4.5 <= magnitude < 6.5:  # Adjusted range for "Medium"
        return "Medium"
    else:
        return "High"  # "High" for anything 6.5 and above

# Apply categorization
data['Intensity'] = data['Magnitude'].apply(categorize_intensity)
y_intensity = data['Intensity']

# Print label distribution for debugging and validation
print("Original Label Distribution:")
print(data['Intensity'].value_counts())

# Augment data for high-magnitude scenarios
high_magnitude_samples = pd.DataFrame({
    'Latitude': np.random.uniform(30, 50, 50),
    'Longitude': np.random.uniform(-130, -100, 50),
    'Depth': np.random.uniform(10, 50, 50),
    'Magnitude': np.random.uniform(6.5, 8, 50)  # Ensure high magnitudes are included
})

# Combine augmented data with original dataset
augmented_data = pd.concat([data, high_magnitude_samples], ignore_index=True)

# Update features and target
X = augmented_data[['Latitude', 'Longitude', 'Depth']]
y_magnitude = augmented_data['Magnitude']
y_intensity = augmented_data['Magnitude'].apply(categorize_intensity)

# Print updated label distribution for validation
print("Augmented Label Distribution:")
print(y_intensity.value_counts())

# Encode intensity levels for classification
y_intensity_encoded = pd.Categorical(y_intensity).codes

# Normalize features
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Split data for regressor and classifier
X_train, X_test, y_magnitude_train, y_magnitude_test = train_test_split(
    X_scaled, y_magnitude, test_size=0.2, random_state=42
)
X_train_cls, X_test_cls, y_intensity_train, y_intensity_test = train_test_split(
    X_scaled, y_intensity_encoded, test_size=0.2, random_state=42
)

# Address class imbalance for the classifier
sample_weights = compute_sample_weight("balanced", y_intensity_train)

# Train XGBoost Regressor
xgb_regressor = XGBRegressor(n_estimators=200, learning_rate=0.1, max_depth=6, random_state=42)
xgb_regressor.fit(X_train, y_magnitude_train)

# Evaluate Regressor
y_magnitude_pred = xgb_regressor.predict(X_test)
mse = mean_squared_error(y_magnitude_test, y_magnitude_pred)
print(f"Mean Squared Error (Regressor): {mse:.2f}")

# Train XGBoost Classifier
xgb_classifier = XGBClassifier(n_estimators=200, learning_rate=0.1, max_depth=6, random_state=42)
xgb_classifier.fit(X_train_cls, y_intensity_train, sample_weight=sample_weights)

# Evaluate Classifier
y_intensity_pred = xgb_classifier.predict(X_test_cls)
accuracy = accuracy_score(y_intensity_test, y_intensity_pred)
print(f"Accuracy (Classifier): {accuracy * 100:.2f}%")

# Generate confusion matrix
conf_matrix = confusion_matrix(y_intensity_test, y_intensity_pred)
print("Confusion Matrix (Classifier):")
print(conf_matrix)

# Save models and scaler
with open('xgb_regressor.pkl', 'wb') as file:
    pickle.dump(xgb_regressor, file)

with open('xgb_classifier.pkl', 'wb') as file:
    pickle.dump(xgb_classifier, file)

with open('scaler.pkl', 'wb') as file:
    pickle.dump(scaler, file)

print("Models and scaler saved successfully.")

