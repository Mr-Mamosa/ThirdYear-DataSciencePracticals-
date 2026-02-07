# P6: Regression and Its Types
# - Implement simple linear regression using a dataset.
# - Explore and interpret the regression model coefficients and goodness-of-fit measures.
# - Extend the analysis to multiple linear regression and assess the impact of additional predictors.

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

# --- 1. Load and Prepare the Data ---
print("--- Loading and Preparing Data ---")
df = pd.read_csv('P6/cars.csv')
# For regression, we'll predict 'Highway mpg'
# Let's select relevant columns and drop rows with missing values
cols_for_regression = [
    'Fuel Information.Highway mpg',
    'Engine Information.Engine Statistics.Horsepower',
    'Engine Information.Engine Statistics.Cylinders',
    'Identification.Year'
]
df_reg = df[cols_for_regression].dropna().copy()
df_reg.rename(columns={
    'Fuel Information.Highway mpg': 'Highway_mpg',
    'Engine Information.Engine Statistics.Horsepower': 'Horsepower',
    'Engine Information.Engine Statistics.Cylinders': 'Cylinders',
    'Identification.Year': 'Year'
}, inplace=True)

print("Data for regression:")
print(df_reg.head())
print("\\n")


# --- 2. Simple Linear Regression ---
print("--- Simple Linear Regression ---")
# Objective: Predict 'Highway_mpg' using 'Horsepower'.

# Define features (X) and target (y)
X_simple = df_reg[['Horsepower']]
y_simple = df_reg['Highway_mpg']

# Split data into training and testing sets
X_train_s, X_test_s, y_train_s, y_test_s = train_test_split(X_simple, y_simple, test_size=0.2, random_state=42)

# Create and train the model
lr_simple = LinearRegression()
lr_simple.fit(X_train_s, y_train_s)

# Make predictions
y_pred_s = lr_simple.predict(X_test_s)

# Evaluate the model
mse_s = mean_squared_error(y_test_s, y_pred_s)
r2_s = r2_score(y_test_s, y_pred_s)

print("Simple Linear Regression Model Evaluation:")
print(f"Mean Squared Error (MSE): {mse_s:.4f}")
print(f"R-squared (R2): {r2_s:.4f}")

# Interpret the model
print(f"\\nCoefficient (slope): {lr_simple.coef_[0]:.4f}")
print(f"Intercept: {lr_simple.intercept_:.4f}")
print("Interpretation: For each one-unit increase in Horsepower, the Highway mpg is expected to decrease by {:.4f}.".format(abs(lr_simple.coef_[0])))
print("\\n")

# Visualize the regression line
plt.figure(figsize=(10, 6))
sns.scatterplot(x=X_test_s['Horsepower'], y=y_test_s, color='blue', label='Actual values')
sns.lineplot(x=X_test_s['Horsepower'], y=y_pred_s, color='red', label='Regression line')
plt.title('Simple Linear Regression: Horsepower vs. Highway mpg')
plt.xlabel('Horsepower')
plt.ylabel('Highway mpg')
plt.legend()
plt.savefig('P6/simple_regression.png') # Save the plot
plt.close()
print("Saved simple linear regression plot to P6/simple_regression.png\\n")


# --- 3. Multiple Linear Regression ---
print("--- Multiple Linear Regression ---")
# Objective: Predict 'Highway_mpg' using 'Horsepower', 'Cylinders', and 'Year'.

# Define features (X) and target (y)
X_multi = df_reg[['Horsepower', 'Cylinders', 'Year']]
y_multi = df_reg['Highway_mpg']

# Split data
X_train_m, X_test_m, y_train_m, y_test_m = train_test_split(X_multi, y_multi, test_size=0.2, random_state=42)

# Create and train the model
lr_multi = LinearRegression()
lr_multi.fit(X_train_m, y_train_m)

# Make predictions
y_pred_m = lr_multi.predict(X_test_m)

# Evaluate the model
mse_m = mean_squared_error(y_test_m, y_pred_m)
r2_m = r2_score(y_test_m, y_pred_m)

print("Multiple Linear Regression Model Evaluation:")
print(f"Mean Squared Error (MSE): {mse_m:.4f}")
print(f"R-squared (R2): {r2_m:.4f}")

# Interpret the model
print("\\nCoefficients:")
for feature, coef in zip(X_multi.columns, lr_multi.coef_):
    print(f"- {feature}: {coef:.4f}")
print(f"Intercept: {lr_multi.intercept_:.4f}")

# Compare with simple regression
print("\\nComparison:")
print(f"R2 score improved from {r2_s:.4f} (simple) to {r2_m:.4f} (multiple).")
print("This suggests that the additional predictors ('Cylinders' and 'Year') have improved the model's ability to explain the variance in 'Highway_mpg'.")
print("\\n")

print("--- Practical 6 execution finished ---")