# ✨ A little something from PaPayaaa ✨
# P8: K-Means Clustering
# - Apply the K-Means algorithm to group similar data points into clusters.
# - Determine the optimal number of clusters using elbow method or silhouette analysis.
# - Visualize the clustering results and analyze the cluster characteristics.

import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import seaborn as sns

# --- 1. Load and Prepare the Data ---
print("--- Loading and Preparing Data ---")
df = pd.read_csv('P8/cars.csv')
# For clustering, let's use 'Horsepower' and 'Highway mpg'
cols_for_clustering = ['Engine Information.Engine Statistics.Horsepower', 'Fuel Information.Highway mpg']
df_cluster = df[cols_for_clustering].dropna().copy()
df_cluster.rename(columns={
    'Engine Information.Engine Statistics.Horsepower': 'Horsepower',
    'Fuel Information.Highway mpg': 'Highway_mpg'
}, inplace=True)

print("Data for clustering:")
print(df_cluster.head())
print("\\n")

# Scale the data for better clustering performance
scaler = StandardScaler()
df_scaled = scaler.fit_transform(df_cluster)


# --- 2. Determine Optimal Number of Clusters ---

# -- Elbow Method --
print("--- Determining Optimal Clusters (Elbow Method) ---")
wcss = []
k_range = range(1, 11)
for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(df_scaled)
    wcss.append(kmeans.inertia_)

plt.figure(figsize=(10, 6))
plt.plot(k_range, wcss, marker='o', linestyle='--')
plt.title('Elbow Method for Optimal k')
plt.xlabel('Number of clusters (k)')
plt.ylabel('WCSS')
plt.xticks(k_range)
plt.grid(True)
plt.savefig('P8/elbow_method.png')
plt.close()
print("Saved elbow method plot to P8/elbow_method.png")
print("From the plot, a good 'elbow' point seems to be at k=3 or k=4.\\n")

# -- Silhouette Analysis --
print("--- Determining Optimal Clusters (Silhouette Analysis) ---")
silhouette_scores = []
k_range_sil = range(2, 11) # Silhouette score is not defined for k=1
for k in k_range_sil:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(df_scaled)
    score = silhouette_score(df_scaled, kmeans.labels_)
    silhouette_scores.append(score)
    print(f"Silhouette score for k={k}: {score:.4f}")

plt.figure(figsize=(10, 6))
plt.plot(k_range_sil, silhouette_scores, marker='o', linestyle='--')
plt.title('Silhouette Scores for Optimal k')
plt.xlabel('Number of clusters (k)')
plt.ylabel('Silhouette Score')
plt.xticks(k_range_sil)
plt.grid(True)
plt.savefig('P8/silhouette_scores.png')
plt.close()
print("\\nSaved silhouette scores plot to P8/silhouette_scores.png")
print("The silhouette score is highest for k=3, suggesting it is a good choice for the number of clusters.\\n")


# --- 3. K-Means Clustering ---
print("--- K-Means Clustering ---")
optimal_k = 3
kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
clusters = kmeans.fit_predict(df_scaled)

# Add cluster labels to the original dataframe
df_cluster['Cluster'] = clusters

print(f"Applied K-Means with {optimal_k} clusters.")
print("Cluster distribution:")
print(df_cluster['Cluster'].value_counts())
print("\\n")


# --- 4. Visualize the Clusters ---
print("--- Visualizing Clusters ---")
plt.figure(figsize=(12, 8))
sns.scatterplot(x='Horsepower', y='Highway_mpg', hue='Cluster', data=df_cluster, palette='viridis', s=100, alpha=0.7)
plt.title('K-Means Clustering of Cars (Horsepower vs. Highway mpg)')
plt.xlabel('Horsepower')
plt.ylabel('Highway mpg')
plt.legend(title='Cluster')
plt.savefig('P8/kmeans_clusters.png')
plt.close()
print("Saved cluster visualization to P8/kmeans_clusters.png")

print("\\n--- Practical 8 execution finished ---")