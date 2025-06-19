# Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
import joblib

# Load data
data_path = '../data/Customer_Data.csv'
df = pd.read_csv(data_path)

# Preview data
print("Data Preview:")
print(df.head())

# Basic info
print("\nData Info:")
print(df.info())

# Check for missing values
print("\nMissing values per column:")
print(df.isnull().sum())

# Fill missing values or drop rows as needed
# (You can customize based on your dataset)
df = df.fillna(method='ffill')  # Simple forward fill example

# Target variable assumption: 'Customer_Status' (replace if different)
print("\nChurn distribution:")
sns.countplot(x='Customer_Status', data=df)
plt.title('Churn Distribution')
plt.show()

# Encode categorical variables (excluding target)
categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
categorical_cols.remove('Customer_Status')  # target variable

label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# Encode target variable: 'Stayed' -> 0, 'Churned' -> 1 (adjust if different)
df['Customer_Status'] = df['Customer_Status'].map({'Stayed': 0, 'Churned': 1})

# Define features and target
X = df.drop('Customer_Status', axis=1)
y = df['Customer_Status']

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train Random Forest model
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Predict on test set
y_pred = rf_model.predict(X_test)

# Evaluate the model
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Plot feature importance
importances = rf_model.feature_importances_
indices = np.argsort(importances)[::-1]

plt.figure(figsize=(12,6))
sns.barplot(x=importances[indices], y=X.columns[indices])
plt.title('Feature Importance')
plt.show()

# Save the trained model
model_path = '../models/churn_rf_model.pkl'
joblib.dump(rf_model, model_path)
print(f"Model saved to {model_path}")
