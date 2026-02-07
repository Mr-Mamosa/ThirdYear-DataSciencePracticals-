# ✨ A little something from PaPayaaa ✨
# P10: Data Visualization and Storytelling
# - Create meaningful visualizations using data visualization tools.
# - Combine multiple visualizations to tell a compelling data story.
# - Present the findings and insights in a clear and concise manner.

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# --- 1. Load and Prepare the Data ---
print("--- Loading and Preparing Data ---")
df = pd.read_csv('P10/cars.csv')
# Let's clean up column names for easier use
df.columns = [col.replace(' ', '_').replace('.', '_') for col in df.columns]
# Rename for simplicity
df.rename(columns={
    'Engine_Information_Engine_Statistics_Horsepower': 'Horsepower',
    'Engine_Information_Engine_Statistics_Cylinders': 'Cylinders',
    'Fuel_Information_Highway_mpg': 'Highway_mpg',
    'Fuel_Information_City_mpg': 'City_mpg',
    'Dimensions_Width': 'Width',
    'Dimensions_Length': 'Length',
    'Dimensions_Height': 'Height',
    'Identification_Make': 'Make',
    'Identification_Year': 'Year'
}, inplace=True)

numerical_cols = ['Horsepower', 'Cylinders', 'Highway_mpg', 'City_mpg', 'Width', 'Length', 'Height', 'Year']
df_vis = df[['Make'] + numerical_cols].dropna()

print("Data for visualization (first 5 rows):")
print(df_vis.head())
print("\\n")


# --- 2. Create Visualizations ---

# -- Story Point 1: What is the overall distribution of Horsepower? --
print("--- Generating Plot 1: Distribution of Horsepower ---")
plt.figure(figsize=(10, 6))
sns.histplot(df_vis['Horsepower'], bins=30, kde=True)
plt.title('Distribution of Horsepower')
plt.xlabel('Horsepower')
plt.ylabel('Frequency')
plt.savefig('P10/1_horsepower_distribution.png')
plt.close()
print("Saved horsepower distribution plot.\\n")

# -- Story Point 2: How does fuel efficiency vary by car make? --
print("--- Generating Plot 2: Fuel Efficiency by Make ---")
# Let's focus on the top 10 most frequent car makes
top_10_makes = df_vis['Make'].value_counts().nlargest(10).index
df_top_makes = df_vis[df_vis['Make'].isin(top_10_makes)]

plt.figure(figsize=(14, 8))
sns.boxplot(x='Make', y='Highway_mpg', data=df_top_makes)
plt.title('Highway MPG by Top 10 Car Makes')
plt.xlabel('Car Make')
plt.ylabel('Highway MPG')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('P10/2_mpg_by_make.png')
plt.close()
print("Saved fuel efficiency by make plot.\\n")

# -- Story Point 3: What are the relationships between numerical features? --
print("--- Generating Plot 3: Correlation Heatmap ---")
plt.figure(figsize=(12, 10))
correlation_matrix = df_vis[numerical_cols].corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Matrix of Car Features')
plt.savefig('P10/3_correlation_heatmap.png')
plt.close()
print("Saved correlation heatmap plot.\\n")

# -- Story Point 4: Deeper dive into key relationships --
print("--- Generating Plot 4: Pair Plot ---")
# A pair plot to visualize relationships between a few key variables
pair_plot_cols = ['Horsepower', 'Highway_mpg', 'Cylinders', 'Width']
sns.pairplot(df_vis[pair_plot_cols])
plt.suptitle('Pair Plot of Key Car Features', y=1.02)
plt.savefig('P10/4_pair_plot.png')
plt.close()
print("Saved pair plot.\\n")

# -- Story Point 5: The trade-off between Horsepower and Fuel Efficiency --
print("--- Generating Plot 5: Horsepower vs. MPG ---")
plt.figure(figsize=(10, 6))
sns.regplot(x='Horsepower', y='Highway_mpg', data=df_vis, scatter_kws={'alpha':0.3})
plt.title('Horsepower vs. Highway MPG')
plt.xlabel('Horsepower')
plt.ylabel('Highway MPG')
plt.savefig('P10/5_hp_vs_mpg.png')
plt.close()
print("Saved horsepower vs. mpg plot.\\n")

print("--- Practical 10 execution finished ---")