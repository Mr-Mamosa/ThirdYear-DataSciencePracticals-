# ✨ A little something from PaPayaaa ✨
# P3: Feature Scaling and Dummification
# - Apply feature-scaling techniques like standardization and normalization to numerical features.
# - Perform feature dummification to convert categorical variables into numerical representations.

import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# --- 1. Load the data ---
print("--- Loading Data ---")
df = pd.read_csv('P3/cars.csv')
# For simplicity, let's work with a subset of columns and handle missing values
columns_to_use = ['Engine Information.Engine Statistics.Horsepower', 'Engine Information.Fuel Type']
df_subset = df[columns_to_use].dropna().copy()
print("Original Data (subset):")
print(df_subset.head())
print("\\n")

# --- 2. Feature Scaling ---

# We'll use the 'Horsepower' column for this demonstration.
hp_col = 'Engine Information.Engine Statistics.Horsepower'

# -- Standardization --
print("--- Standardization ---")
# StandardScaler standardizes features by removing the mean and scaling to unit variance.
scaler_std = StandardScaler()
# The output of scaler is a numpy array, so we reshape it and add it as a new column
df_subset['Horsepower_Standardized'] = scaler_std.fit_transform(df_subset[[hp_col]])
print("Data after Standardization:")
print(df_subset[['Horsepower_Standardized', hp_col]].head())
print(f"Mean of standardized horsepower: {df_subset['Horsepower_Standardized'].mean():.2f}")
print(f"Standard Deviation of standardized horsepower: {df_subset['Horsepower_Standardized'].std():.2f}")
print("\\n")


# -- Normalization --
print("--- Normalization ---")
# MinMaxScaler scales and translates each feature individually such that it is in the given range on the training set, e.g., between zero and one.
scaler_norm = MinMaxScaler()
df_subset['Horsepower_Normalized'] = scaler_norm.fit_transform(df_subset[[hp_col]])
print("Data after Normalization:")
print(df_subset[['Horsepower_Normalized', hp_col]].head())
print(f"Min of normalized horsepower: {df_subset['Horsepower_Normalized'].min():.2f}")
print(f"Max of normalized horsepower: {df_subset['Horsepower_Normalized'].max():.2f}")
print("\\n")


# --- 3. Dummification ---

print("--- Dummification ---")
# Dummification is the process of converting categorical variables into dummy or indicator variables.
fuel_type_col = 'Engine Information.Fuel Type'
print(f"Original unique values in '{fuel_type_col}':")
print(df_subset[fuel_type_col].unique())
print("\\n")

# Use pd.get_dummies to convert the 'Fuel Type' column
dummies = pd.get_dummies(df_subset[fuel_type_col], prefix='FuelType')

# Concatenate the new dummy variables with the original dataframe
df_dummified = pd.concat([df_subset, dummies], axis=1)

print("Data after Dummification:")
print(df_dummified.head())
print("\\n")

# We can now drop the original 'Fuel Type' column if we want
# df_dummified.drop(fuel_type_col, axis=1, inplace=True)
# print("Dataframe after dropping the original categorical column:")
# print(df_dummified.head())

print("--- Practical 3 execution finished ---")