# ✨ A little something from PaPayaaa ✨
# P9: Principal Component Analysis (PCA)
# - Perform PCA on a dataset to reduce dimensionality.
# - Evaluate the explained variance and select the appropriate number of principal components.
# - Visualize the data in the reduced-dimensional space.

import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np

# --- 1. Load and Prepare the Data ---
print("--- Loading and Preparing Data ---")
df = pd.read_csv('P9/cars.csv')
# For PCA, we'll use a set of numerical features
numerical_cols = [
    'Engine Information.Engine Statistics.Horsepower',
    'Engine Information.Engine Statistics.Cylinders',
    'Fuel Information.Highway mpg',
    'Fuel Information.City mpg',
    'Dimensions.Width',
    'Dimensions.Length',
    'Dimensions.Height',
    'Identification.Year'
]
df_pca = df[numerical_cols].dropna().copy()

print("Data for PCA (first 5 rows):")
print(df_pca.head())
print("\\n")

# Scale the data before applying PCA
scaler = StandardScaler()
df_scaled = scaler.fit_transform(df_pca)


# --- 2. Perform PCA ---
print("--- Performing PCA ---")
# We'll start by fitting PCA without specifying the number of components
# to see how much variance is explained by each.
pca = PCA()
pca.fit(df_scaled)

# Explained variance ratio
explained_variance = pca.explained_variance_ratio_
print("Explained variance ratio of each principal component:")
for i, var in enumerate(explained_variance):
    print(f"  PC{i+1}: {var:.4f}")
print("\\n")


# --- 3. Select Number of Components ---
print("--- Selecting Number of Components ---")
# We'll plot the cumulative explained variance to decide how many components to keep.
cumulative_variance = np.cumsum(explained_variance)

plt.figure(figsize=(10, 6))
plt.plot(range(1, len(cumulative_variance) + 1), cumulative_variance, marker='o', linestyle='--')
plt.title('Cumulative Explained Variance')
plt.xlabel('Number of Principal Components')
plt.ylabel('Cumulative Explained Variance')
plt.axhline(y=0.95, color='r', linestyle='--', label='95% Explained Variance')
plt.legend()
plt.grid(True)
plt.savefig('P9/cumulative_variance.png')
plt.close()
print("Saved cumulative explained variance plot to P9/cumulative_variance.png")

# Find the number of components to explain 95% of variance
n_components_95 = np.where(cumulative_variance >= 0.95)[0][0] + 1
print(f"Number of components to explain 95% of variance: {n_components_95}\\n")


# --- 4. Visualize the Data in Reduced-Dimensional Space ---
print("--- Visualizing Data with PCA ---")
# We'll use the first two principal components to visualize the data.
pca_2 = PCA(n_components=2)
principal_components = pca_2.fit_transform(df_scaled)

df_pca_2d = pd.DataFrame(data=principal_components, columns=['PC1', 'PC2'])

plt.figure(figsize=(12, 8))
plt.scatter(df_pca_2d['PC1'], df_pca_2d['PC2'], alpha=0.5)
plt.title('2D PCA of Cars Dataset')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.grid(True)
plt.savefig('P9/pca_2d_visualization.png')
plt.close()
print("Saved 2D PCA visualization plot to P9/pca_2d_visualization.png")
print("This plot shows the data projected onto the first two principal components.")
print("The spread of the data along each axis corresponds to the variance explained by that component.")
print("\\n")

print("--- Practical 9 execution finished ---")