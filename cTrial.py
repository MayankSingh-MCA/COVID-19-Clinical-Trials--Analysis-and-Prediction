# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.impute import SimpleImputer

# Load the dataset
file_path = r'C:\Users\bdrin\Desktop\COVID\COVID clinical trials.csv'  # Update with your file path
data = pd.read_csv(file_path)

# Display the first few rows of the dataset
print("Dataset Preview:")
print(data.head())

# ---------------------- Data Cleaning ----------------------

# Drop irrelevant or duplicate columns
data.drop(columns=['NCT Number', 'URL', 'Study Documents'], inplace=True)

# Check for missing values
missing_data = data.isnull().sum() / len(data) * 100
print("\nMissing Data Percentage:")
print(missing_data)

# Impute missing values
# For categorical features, use mode
cat_features = data.select_dtypes(include=['object']).columns
imputer_cat = SimpleImputer(strategy='most_frequent')
data[cat_features] = imputer_cat.fit_transform(data[cat_features])

# For numerical features, use median
num_features = data.select_dtypes(include=['float64', 'int64']).columns
imputer_num = SimpleImputer(strategy='median')
data[num_features] = imputer_num.fit_transform(data[num_features])

# Recheck for missing values
print("\nMissing Data After Imputation:")
print(data.isnull().sum())

# ---------------------- Feature Engineering ----------------------

# Extract 'Country' from 'Locations'
data['Country'] = data['Locations'].str.extract(r',\s*([^,]+)$')[0]

# Create a new feature for trial duration
data['Start Date'] = pd.to_datetime(data['Start Date'], errors='coerce')
data['Completion Date'] = pd.to_datetime(data['Completion Date'], errors='coerce')
data['Trial Duration'] = (data['Completion Date'] - data['Start Date']).dt.days
data['Trial Duration'] = data['Trial Duration'].fillna(0)  # Fill NaN with 0

# ---------------------- Exploratory Data Analysis (EDA) ----------------------

# Descriptive statistics
print("\nDescriptive Statistics:")
print(data.describe(include='all'))

# Visualizations
plt.figure(figsize=(10, 6))
sns.countplot(data=data, x='Status', order=data['Status'].value_counts().index)
plt.title('Distribution of Trial Status')
plt.xticks(rotation=45)
plt.show()

plt.figure(figsize=(10, 6))
sns.countplot(data=data, x='Phases', order=data['Phases'].value_counts().index)
plt.title('Distribution of Trial Phases')
plt.xticks(rotation=45)
plt.show()

plt.figure(figsize=(10, 6))
sns.histplot(data['Enrollment'], bins=30, kde=True)
plt.title('Distribution of Enrollment')
plt.xlabel('Enrollment')
plt.ylabel('Frequency')
plt.show()

# Correlation Analysis
numeric_data = data.select_dtypes(include=['float64', 'int64'])
correlation_matrix = numeric_data.corr()

plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap='coolwarm', cbar=True)
plt.title('Correlation Matrix')
plt.show()

# ---------------------- Predictive Modeling ----------------------

# Prepare the data
# Select features and target variable
X = data[['Enrollment', 'Trial Duration']]  # Add more features if needed
y = data['Status'].apply(lambda x: 1 if x == 'Completed' else 0)  # Binary classification

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Random Forest Classifier
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# ---------------------- Model Evaluation ----------------------

# Classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Confusion matrix
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# ---------------------- Feature Importance ----------------------

importances = model.feature_importances_
feature_names = X.columns
indices = np.argsort(importances)[::-1]

plt.figure(figsize=(10, 6))
plt.title("Feature Importances")
plt.bar(range(X.shape[1]), importances[indices], align="center")
plt.xticks(range(X.shape[1]), feature_names[indices], rotation=45)
plt.xlim([-1, X.shape[1]])
plt.show()
