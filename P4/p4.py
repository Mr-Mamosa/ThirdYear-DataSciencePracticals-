# ✨ A little something from PaPayaaa ✨
# P4: Hypothesis Testing
# - Formulate null and alternative hypotheses for a given problem.
# - Conduct a hypothesis test using appropriate statistical tests (e.g., t-test, chi-square test).
# - Interpret the results and draw conclusions based on the test outcomes.

import pandas as pd
from scipy.stats import ttest_ind, chi2_contingency

# --- 1. Load the data ---
print("--- Loading Data ---")
df = pd.read_csv('P4/cars.csv')
# For simplicity, let's work with a subset of columns and handle missing values
columns_to_use = ['Engine Information.Engine Statistics.Horsepower', 'Engine Information.Fuel Type', 'Identification.Make', 'Engine Information.Engine Statistics.Cylinders']
df_subset = df[columns_to_use].dropna().copy()
print("Original Data (subset):")
print(df_subset.head())
print("\\n")

# --- 2. T-test ---
print("--- Independent Samples T-test ---")
# Objective: To test if there is a significant difference in horsepower between gasoline and diesel cars.

# Hypotheses:
# H0 (Null Hypothesis): There is no significant difference in the mean horsepower between gasoline and diesel cars.
# H1 (Alternative Hypothesis): There is a significant difference in the mean horsepower between gasoline and diesel cars.

fuel_type_col = 'Engine Information.Fuel Type'
hp_col = 'Engine Information.Engine Statistics.Horsepower'

# Separate the horsepower data for gasoline and diesel cars
gasoline_hp = df_subset[df_subset[fuel_type_col] == 'Gasoline'][hp_col]
diesel_hp = df_subset[df_subset[fuel_type_col] == 'Diesel'][hp_col]

# Perform the t-test
t_stat, p_value = ttest_ind(gasoline_hp, diesel_hp, equal_var=False) # Welch's t-test

print(f"T-statistic: {t_stat:.4f}")
print(f"P-value: {p_value:.4f}")

# Interpretation
alpha = 0.05
if p_value < alpha:
    print("Conclusion: Reject the null hypothesis.")
    print("There is a statistically significant difference in horsepower between gasoline and diesel cars.")
else:
    print("Conclusion: Fail to reject the null hypothesis.")
    print("There is no statistically significant difference in horsepower between gasoline and diesel cars.")
print("\\n")


# --- 3. Chi-square test ---
print("--- Chi-square Test of Independence ---")
# Objective: To test if there is a significant association between the car's make and its number of cylinders.

# Hypotheses:
# H0 (Null Hypothesis): There is no association between car make and the number of cylinders.
# H1 (Alternative Hypothesis): There is an association between car make and the number of cylinders.

make_col = 'Identification.Make'
cylinders_col = 'Engine Information.Engine Statistics.Cylinders'

# To make the test meaningful, we'll consider a subset of car makes.
top_makes = df_subset[make_col].value_counts().nlargest(5).index
df_filtered_makes = df_subset[df_subset[make_col].isin(top_makes)]

# Create a contingency table
contingency_table = pd.crosstab(df_filtered_makes[make_col], df_filtered_makes[cylinders_col])
print("Contingency Table (Make vs. Cylinders):")
print(contingency_table)
print("\\n")

# Perform the chi-square test
chi2, p_value_chi, dof, expected = chi2_contingency(contingency_table)

print(f"Chi-square statistic: {chi2:.4f}")
print(f"P-value: {p_value_chi:.4f}")
print(f"Degrees of freedom: {dof}")
# print("Expected frequencies:\\n", expected)

# Interpretation
if p_value_chi < alpha:
    print("Conclusion: Reject the null hypothesis.")
    print("There is a statistically significant association between car make and the number of cylinders.")
else:
    print("Conclusion: Fail to reject the null hypothesis.")
    print("There is no statistically significant association between car make and the number of cylinders.")
print("\\n")

print("--- Practical 4 execution finished ---")
