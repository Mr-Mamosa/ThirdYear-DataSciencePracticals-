# ✨ A little something from PaPayaaa ✨
# P2: Data Frames and Basic Data Pre-processing
# - Read data from CSV and JSON files into a data frame.
# - Perform basic data pre-processing tasks such as handling missing values and outliers.
# - Manipulate and transform data using functions like filtering, sorting, and grouping.

import pandas as pd
import numpy as np

# --- 1. Read data from CSV and JSON files ---

# Read from CSV
print("--- Reading from CSV ---")
csv_file_path = 'P2/cars.csv'
df_csv = pd.read_csv(csv_file_path)
print("First 5 rows of the CSV data:")
print(df_csv.head())
print("\\n")

# Read from JSON
print("--- Reading from JSON ---")
json_file_path = 'P2/dummy.json'
df_json = pd.read_json(json_file_path)
print("Data from the JSON file:")
print(df_json)
print("\\n")


# --- 2. Handle Missing Values ---

print("--- Handling Missing Values ---")
# Create a copy to keep the original dataframe intact
df = df_csv.copy()

# Check for missing values
print("Missing values before handling:")
print(df.isnull().sum())
print("\\n")

# For simplicity, we'll focus on a few columns. Let's look at 'Engine Information.Engine Statistics.Horsepower'.
# It has missing values. Let's fill them with the mean.
horsepower_col = 'Engine Information.Engine Statistics.Horsepower'
if horsepower_col in df.columns:
    mean_hp = df[horsepower_col].mean()
    df[horsepower_col].fillna(mean_hp, inplace=True)
    print(f"Filled missing values in '{horsepower_col}' with the mean ({mean_hp:.2f}).")
else:
    print(f"Column '{horsepower_col}' not found. Skipping missing value handling for it.")

# Let's check another column, for example, 'Engine Information.Fuel Type'.
# If it has missing values, we can fill with the mode.
fuel_type_col = 'Engine Information.Fuel Type'
if fuel_type_col in df.columns and df[fuel_type_col].isnull().any():
    mode_fuel = df[fuel_type_col].mode()[0]
    df[fuel_type_col].fillna(mode_fuel, inplace=True)
    print(f"Filled missing values in '{fuel_type_col}' with the mode ('{mode_fuel}').")

# Alternatively, we could drop rows with any missing values
# df.dropna(inplace=True)
# print("Dropped rows with missing values.")

print("\\nMissing values after handling:")
print(df[horsepower_col].isnull().sum())
print("\\n")


# --- 3. Handle Outliers ---

print("--- Handling Outliers ---")
# We'll use the 'Identification.Year' column for this demonstration.
year_col = 'Identification.Year'

if year_col in df.columns:
    Q1 = df[year_col].quantile(0.25)
    Q3 = df[year_col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    print(f"For '{year_col}':")
    print(f"Q1: {Q1}, Q3: {Q3}, IQR: {IQR}")
    print(f"Lower bound for outliers: {lower_bound}")
    print(f"Upper bound for outliers: {upper_bound}")

    # Find outliers
    outliers = df[(df[year_col] < lower_bound) | (df[year_col] > upper_bound)]
    print(f"\\nNumber of outliers detected in '{year_col}': {len(outliers)}")

    # Handling outliers by capping them
    df[year_col] = np.where(df[year_col] < lower_bound, lower_bound, df[year_col])
    df[year_col] = np.where(df[year_col] > upper_bound, upper_bound, df[year_col])

    print("Outliers have been capped to the lower and upper bounds.")
    print(f"Min year after capping: {df[year_col].min()}")
    print(f"Max year after capping: {df[year_col].max()}")
    print("\\n")
else:
    print(f"Column '{year_col}' not found. Skipping outlier handling.")


# --- 4. Data Manipulation ---

print("--- Data Manipulation ---")

# Filtering data
print("Filtering: Cars with more than 4 cylinders")
cylinders_col = 'Engine Information.Engine Statistics.Cylinders'
if cylinders_col in df.columns:
    high_cylinder_cars = df[df[cylinders_col] > 4]
    print(high_cylinder_cars[['Identification.Make', cylinders_col]].head())
else:
    print(f"Column '{cylinders_col}' not found. Skipping filtering.")
print("\\n")

# Sorting data
print("Sorting: Cars sorted by year (descending)")
sorted_cars = df.sort_values(by=year_col, ascending=False)
print(sorted_cars[['Identification.Make', year_col]].head())
print("\\n")

# Grouping data
print("Grouping: Average horsepower by car make")
if horsepower_col in df.columns and 'Identification.Make' in df.columns:
    avg_hp_by_make = df.groupby('Identification.Make')[horsepower_col].mean().reset_index()
    avg_hp_by_make.columns = ['Make', 'Average Horsepower']
    print(avg_hp_by_make.head())
else:
    print("Required columns for grouping not found. Skipping.")

print("\\n--- Practical 2 execution finished ---")