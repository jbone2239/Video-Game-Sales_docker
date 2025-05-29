import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score
import joblib

# file path
file_path = r'C:\Users\allib\OneDrive\Desktop\MS Data Science\ANA680\Week4\final project\vgsales.csv'

# load the CSV file
df = pd.read_csv(file_path)

# preview my data
print("First 5 rows of the dataset:")
print(df.head())

# basic data exploration
print("\nDataset info:")
print(df.info())

print("\nDescriptive statistics:")
print(df.describe())

print("\nMissing values per column:")
print(df.isnull().sum())

print("\nNumber of unique values per column:")
print(df.nunique())

# drop rows with missing required fields
df = df.dropna(subset=['Platform', 'Genre', 'Publisher', 'Year', 'NA_Sales', 'EU_Sales', 'JP_Sales'])


# create the target variable 'Top_Region'
def get_top_region(row):
    return max(['NA_Sales', 'EU_Sales', 'JP_Sales'], key=lambda region: row[region])

df['Top_Region'] = df.apply(get_top_region, axis=1)

# shouldn't use future sales to predict
df = df.drop(columns=['NA_Sales', 'EU_Sales', 'JP_Sales', 'Other_Sales', 'Global_Sales'])

# prepare feature and target sets
X = df[['Platform', 'Genre', 'Publisher', 'Year']]
y = df['Top_Region']

# one-hot encode categorical features
X_encoded = pd.get_dummies(X, columns=['Platform', 'Genre', 'Publisher'])

# save feature column names to CSV for Flask input mapping
X_encoded.columns.to_series().to_csv('model_columns.csv', index=False, header=False)

# encode target labels
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# train-test split
X_train, X_val, y_train, y_val = train_test_split(
    X_encoded, y_encoded, test_size=0.3, random_state=42, stratify=y_encoded)

# train the Random Forest model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# evaluate model performance
y_val_pred = model.predict(X_val)
print("Validation Accuracy:", accuracy_score(y_val, y_val_pred))
print("\nClassification Report:")
print(classification_report(y_val, y_val_pred, target_names=le.classes_))

# save the model and label encoder
joblib.dump(model, 'vgsales_model.pkl')
joblib.dump(le, 'label_encoder.pkl')