import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# Load the data
df = pd.read_excel("/Users/fangxinyu/Desktop/Household meter raw data.xlsx")

# Step 1: Filter out days with abnormal daily consumption
# Calculate daily consumption (sum of hourly values from column 2 onwards)
daily_consumption = df.iloc[:, 2:].sum(axis=1)

# Keep only days with consumption between 20L and 1000L
filtered_df = df[(daily_consumption >= 20) & (daily_consumption <= 1000)]

# Step 2: Filter out households with insufficient data
# Keep only households (IDs) with more than 30 days of data
filtered_df_grouped = filtered_df.groupby('ID').filter(lambda x: len(x) > 30)

# Step 3: Prepare data for clustering
# Fill missing values with 0
df_prepared = filtered_df_grouped.fillna(0)

# Create pivot table with Date as index (excluding ID column)
pivot_df = df_prepared.drop('ID', axis=1).set_index('Date')

# Step 4: Perform K-means clustering
num_clusters = 3  # Adjust this based on your analysis needs

# Initialize and fit K-means model
kmeans = KMeans(n_clusters=num_clusters, random_state=0)
clusters = kmeans.fit_predict(pivot_df)

# Add cluster labels to the dataframe
pivot_df['Cluster'] = clusters

# Step 5: Analyze clustering results
# Get cluster centers (typical daily profiles for each cluster)
cluster_centers = kmeans.cluster_centers_

# Count total number of data points
total_count = len(clusters)

# Count occurrences in each cluster
cluster_counts = np.bincount(clusters)

# Step 6: Filter out small clusters (less than 5% of total data)
threshold_count = 0.05 * total_count

# Identify clusters that meet the minimum size requirement
valid_clusters = np.where(cluster_counts >= threshold_count)[0]

# Keep only data from valid clusters
valid_cluster_data = pivot_df[pivot_df['Cluster'].isin(valid_clusters)]

# Step 7: Optional - Calculate clustering quality metrics
if len(valid_clusters) > 1:
    # Calculate silhouette score for valid clusters only
    valid_data_features = valid_cluster_data.iloc[:, :-1]  # Exclude 'Cluster' column
    valid_labels = valid_cluster_data['Cluster']
    silhouette_avg = silhouette_score(valid_data_features, valid_labels)
    print(f"Silhouette Score: {silhouette_avg:.3f}")

# Display results
print(f"Total number of days in dataset: {total_count}")
print(f"Number of clusters: {num_clusters}")
print(f"Cluster distribution:")
for i in range(num_clusters):
    count = cluster_counts[i]
    percentage = (count / total_count) * 100
    status = "Valid" if i in valid_clusters else "Excluded (too small)"
    print(f"  Cluster {i}: {count} days ({percentage:.1f}%) - {status}")

print(f"\nNumber of valid clusters: {len(valid_clusters)}")
print(f"Total days in valid clusters: {len(valid_cluster_data)}")

# Optional: Visualize cluster centers (typical daily load profiles)
if len(valid_clusters) > 0:
    import matplotlib.pyplot as plt
    
    hours = list(range(24))
    plt.figure(figsize=(12, 6))
    
    for cluster_id in valid_clusters:
        plt.plot(hours, cluster_centers[cluster_id], 
                marker='o', label=f'Cluster {cluster_id}')
    
    plt.xlabel('Hour of Day')
    plt.ylabel('Electricity Consumption (kWh)')
    plt.title('Typical Daily Load Profiles by Cluster')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xticks(hours)
    plt.show()
