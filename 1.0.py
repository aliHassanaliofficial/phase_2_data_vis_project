import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# Load the dataset
dataset_path = "./student_habits_performance.csv"
df = pd.read_csv(dataset_path)

# Display the first few rows and columns of the dataset
print("Initial Data Preview:")
print(df.head())
print("\nInitial Columns:", df.columns.tolist())

# Remove rows with missing values and duplicates
df = df.dropna()
df = df.drop_duplicates()

# Create a binary target column: 1 = pass (score >= 50), 0 = fail
df['target'] = df['exam_score'].apply(lambda x: 1 if x >= 50 else 0)

# Drop unhelpful columns
df = df.drop(['student_id', 'exam_score'], axis=1)

# Separate target
y = df['target']
X = df.drop('target', axis=1)

# One-hot encode categorical features
X = pd.get_dummies(X, drop_first=True)

print("\nProcessed Columns (after encoding):", X.columns.tolist())

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train SVM model
model = svm.SVC(kernel='linear')
model.fit(X_train, y_train)

# Evaluate model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"\nModel Accuracy: {accuracy * 100:.2f}%")

# Save the model
joblib.dump(model, 'svm_model.pkl')

# Load and re-evaluate the model
loaded_model = joblib.load('svm_model.pkl')
loaded_y_pred = loaded_model.predict(X_test)
loaded_accuracy = accuracy_score(y_test, loaded_y_pred)
print(f"Loaded Model Accuracy: {loaded_accuracy * 100:.2f}%")

# Save predictions
predictions_df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
predictions_df.to_csv('predictions.csv', index=False)

# Load and display predictions
loaded_predictions_df = pd.read_csv('predictions.csv')
print("\nLoaded Predictions:")
print(loaded_predictions_df.head())

# Save the processed dataset
processed_df = X.copy()
processed_df['target'] = y.values
processed_df.to_csv('processed_dataset.csv', index=False)

# --- Visualization ---

sns.set(style="whitegrid")

# Pairplot
sns.pairplot(processed_df, hue='target')
plt.title('Pairplot of Features')
plt.show()

# Correlation Heatmap
plt.figure(figsize=(12, 8))
correlation_matrix = processed_df.corr()
sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap='coolwarm', square=True)
plt.title('Correlation Heatmap')
plt.show()

# Target distribution
plt.figure(figsize=(8, 6))
sns.countplot(x='target', data=processed_df)
plt.title('Distribution of Target Variable (Pass/Fail)')
plt.xlabel('Target (1=Pass, 0=Fail)')
plt.ylabel('Count')
plt.show()


exit(0)