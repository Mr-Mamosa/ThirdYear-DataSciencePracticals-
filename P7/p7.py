# ✨ A little something from PaPayaaa ✨
# P7: Logistic Regression and Decision Tree
# - Build a logistic regression model to predict a binary outcome.
# - Evaluate the model's performance using classification metrics (e.g., accuracy, precision, recall).
# - Construct a decision tree model and interpret the decision rules for classification.

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
import matplotlib.pyplot as plt

# --- 1. Load and Prepare the Data ---
print("--- Loading and Preparing Data ---")
df = pd.read_csv('P7/cars.csv')
# Create a binary target variable 'Fuel_Efficient'
cols_to_use = [
    'Fuel Information.Highway mpg',
    'Engine Information.Engine Statistics.Horsepower',
    'Engine Information.Engine Statistics.Cylinders',
    'Identification.Year'
]
df_class = df[cols_to_use].dropna().copy()
df_class.rename(columns={
    'Fuel Information.Highway mpg': 'Highway_mpg',
    'Engine Information.Engine Statistics.Horsepower': 'Horsepower',
    'Engine Information.Engine Statistics.Cylinders': 'Cylinders',
    'Identification.Year': 'Year'
}, inplace=True)

# Create binary target variable
median_mpg = df_class['Highway_mpg'].median()
df_class['Fuel_Efficient'] = (df_class['Highway_mpg'] > median_mpg).astype(int)

print("Data for classification with binary target 'Fuel_Efficient':")
print(df_class.head())
print("\\n")


# --- 2. Logistic Regression ---
print("--- Logistic Regression ---")
# Define features (X) and target (y)
features = ['Horsepower', 'Cylinders', 'Year']
X = df_class[features]
y = df_class['Fuel_Efficient']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
log_reg = LogisticRegression(random_state=42)
log_reg.fit(X_train, y_train)

# Make predictions
y_pred_lr = log_reg.predict(X_test)

# Evaluate the model
accuracy_lr = accuracy_score(y_test, y_pred_lr)
precision_lr = precision_score(y_test, y_pred_lr)
recall_lr = recall_score(y_test, y_pred_lr)
conf_matrix_lr = confusion_matrix(y_test, y_pred_lr)

print("Logistic Regression Model Evaluation:")
print(f"Accuracy: {accuracy_lr:.4f}")
print(f"Precision: {precision_lr:.4f}")
print(f"Recall: {recall_lr:.4f}")
print("Confusion Matrix:\\n", conf_matrix_lr)
print("\\n")


# --- 3. Decision Tree ---
print("--- Decision Tree ---")
# Train the model
# We use max_depth to prevent overfitting and make the tree interpretable
dtree = DecisionTreeClassifier(max_depth=3, random_state=42)
dtree.fit(X_train, y_train)

# Make predictions
y_pred_dt = dtree.predict(X_test)

# Evaluate the model
accuracy_dt = accuracy_score(y_test, y_pred_dt)
precision_dt = precision_score(y_test, y_pred_dt)
recall_dt = recall_score(y_test, y_pred_dt)
conf_matrix_dt = confusion_matrix(y_test, y_pred_dt)

print("Decision Tree Model Evaluation:")
print(f"Accuracy: {accuracy_dt:.4f}")
print(f"Precision: {precision_dt:.4f}")
print(f"Recall: {recall_dt:.4f}")
print("Confusion Matrix:\\n", conf_matrix_dt)
print("\\n")

# Visualize the decision tree
print("Visualizing Decision Tree...")
plt.figure(figsize=(20,10))
plot_tree(dtree,
          feature_names=features,
          class_names=['Not Fuel Efficient', 'Fuel Efficient'],
          filled=True,
          rounded=True,
          fontsize=10)
plt.title("Decision Tree for Fuel Efficiency")
plt.savefig('P7/decision_tree.png') # Save the plot
plt.close()
print("Saved decision tree plot to P7/decision_tree.png\\n")

print("--- Practical 7 execution finished ---")