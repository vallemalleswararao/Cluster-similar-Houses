from sklearn.cluster import KMeans
import pandas as pd

# Assuming X_scaled is your scaled data from the previous code
# Let's assume the optimal number of clusters is k_optimal

kmeans = KMeans(n_clusters=k_optimal, init='k-means++', max_iter=300, n_init=10, random_state=0)
cluster_labels = kmeans.fit_predict(X_scaled)

# Add the cluster labels to your original DataFrame
data_with_clusters = pd.DataFrame(data=X_scaled, columns=X.columns)  # Assuming your data is stored in a DataFrame X
data_with_clusters['Cluster'] = cluster_labels

# Now, data_with_clusters contains a new 'Cluster' column with cluster assignments
