# ✨ A little something from PaPayaaa ✨
# P5: ANOVA (Analysis of Variance)
# - Perform one-way ANOVA to compare means across multiple groups.
# - Conduct post-hoc tests to identify significant differences between group means.

import pandas as pd
from scipy.stats import f_oneway
from statsmodels.stats.multicomp import pairwise_tukeyhsd

# --- 1. Load the data ---
print("--- Loading Data ---")
df = pd.read_csv('P5/cars.csv')
# For this practical, we'll look at Horsepower and Cylinders
columns_to_use = ['Engine Information.Engine Statistics.Horsepower', 'Engine Information.Engine Statistics.Cylinders']
df_subset = df[columns_to_use].dropna().copy()
print("Original Data (subset):")
print(df_subset.head())
print("\\n")


# --- 2. One-way ANOVA ---
print("--- One-way ANOVA ---")
# Objective: To test if there is a significant difference in the mean horsepower of cars with different numbers of cylinders.

# Hypotheses:
# H0 (Null Hypothesis): There is no significant difference in the mean horsepower of cars with different numbers of cylinders.
# H1 (Alternative Hypothesis): There is a significant difference in the mean horsepower for at least one number of cylinders.

cylinders_col = 'Engine Information.Engine Statistics.Cylinders'
hp_col = 'Engine Information.Engine Statistics.Horsepower'

# Get the unique cylinder groups
cylinder_groups = df_subset[cylinders_col].unique()

# Create a list of horsepower values for each cylinder group
grouped_data = [df_subset[hp_col][df_subset[cylinders_col] == c] for c in cylinder_groups]

# Perform the one-way ANOVA
f_stat, p_value = f_oneway(*grouped_data)

print(f"F-statistic: {f_stat:.4f}")
print(f"P-value: {p_value:.4f}")

# Interpretation
alpha = 0.05
if p_value < alpha:
    print("Conclusion: Reject the null hypothesis.")
    print("There is a statistically significant difference in mean horsepower among cars with different numbers of cylinders.")
    print("\\n--- Performing Post-hoc Test (Tukey's HSD) ---")
    
    # --- 3. Post-hoc Test (Tukey's HSD) ---
    tukey = pairwise_tukeyhsd(endog=df_subset[hp_col],
                              groups=df_subset[cylinders_col],
                              alpha=alpha)
    
    print(tukey)
    
    # To make it more readable, we can convert the results to a DataFrame
    df_tukey = pd.DataFrame(data=tukey._results_table.data[1:], columns=tukey._results_table.data[0])
    
    print("\\nSignificant differences found between the following groups:")
    print(df_tukey[df_tukey['reject'] == True])

else:
    print("Conclusion: Fail to reject the null hypothesis.")
    print("There is no statistically significant difference in mean horsepower among cars with different numbers of cylinders.")

print("\\n--- Practical 5 execution finished ---")