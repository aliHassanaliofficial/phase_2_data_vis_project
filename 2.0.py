import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix

# Load dataset
df = pd.read_csv("./student_habits_performance.csv")

# Clean data
df.dropna(inplace=True)
df.drop_duplicates(inplace=True)

# Convert to performance labels
df['Performance'] = pd.cut(df['exam_score'], bins=[0, 50, 75, 100], labels=['Low', 'Medium', 'High'])

# Split features and labels
X = df.drop(['student_id', 'exam_score', 'Performance'], axis=1)
y = df['Performance']

# Encode categorical
X = pd.get_dummies(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# EDA
numeric_cols = X_train.select_dtypes(include=np.number).columns
for col in numeric_cols:
    sns.histplot(X_train[col], kde=True)
    plt.title(f"Distribution of {col}")
    plt.show()

# Normalize
scaler = MinMaxScaler()
X_train.loc[:, numeric_cols] = scaler.fit_transform(X_train[numeric_cols])
X_test.loc[:, numeric_cols] = scaler.transform(X_test[numeric_cols])

# Fuzzy feature creation
def triangular_membership(x, a, b, c):
    return np.maximum(np.minimum((x - a) / (b - a + 1e-9), (c - x) / (c - b + 1e-9)), 0)

fuzzy_features = ['study_hours_per_day', 'sleep_hours', 'social_media_hours']
for feature in fuzzy_features:
    if feature in X_train.columns:
        for label, (a, b, c) in {
            'low': (0, 0.25, 0.5),
            'medium': (0.25, 0.5, 0.75),
            'high': (0.5, 0.75, 1.0)
        }.items():
            X_train[f'{feature}_{label}'] = triangular_membership(X_train[feature], a, b, c)
            X_test[f'{feature}_{label}'] = triangular_membership(X_test[feature], a, b, c)

# Hill-climbing search
def train_and_evaluate(depth):
    clf = DecisionTreeClassifier(max_depth=depth, random_state=42)
    clf.fit(X_train, y_train)
    preds = clf.predict(X_test)
    return accuracy_score(y_test, preds), clf

best_depth = 1
best_score, best_model = train_and_evaluate(best_depth)
for d in range(2, 11):
    score, model = train_and_evaluate(d)
    if score > best_score:
        best_score = score
        best_depth = d
        best_model = model
    else:
        break

print(f"Hill-Climbing Best max_depth: {best_depth}, Accuracy: {best_score:.4f}")

# Grid Search
print("Grid Search Results:")
for d in range(1, 11):
    score, _ = train_and_evaluate(d)
    print(f"Depth {d}: Accuracy = {score:.4f}")

# Visualize decision tree
plt.figure(figsize=(20, 10))
plot_tree(best_model, filled=True, feature_names=X_train.columns, class_names=best_model.classes_)
plt.title("Final Decision Tree")
plt.show()

# Final evaluation
final_preds = best_model.predict(X_test)
print("Final Evaluation:")
print("Accuracy:", accuracy_score(y_test, final_preds))
print("Precision:", precision_score(y_test, final_preds, average='weighted'))
print("Recall:", recall_score(y_test, final_preds, average='weighted'))
print("Confusion Matrix:\n", confusion_matrix(y_test, final_preds))
