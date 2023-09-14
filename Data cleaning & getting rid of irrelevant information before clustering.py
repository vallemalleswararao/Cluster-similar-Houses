import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns

# Load your dataset (replace 'data.csv' with your data file)
data = pd.read_csv('data.csv')

# 1. Handling Missing Data
data.dropna(inplace=True)  # Remove rows with missing values

# 2. Encode Categorical Variables
# If you have categorical variables, use Label Encoding or One-Hot Encoding
# Example using Label Encoding:
label_encoder = LabelEncoder()
data['categorical_column'] = label_encoder.fit_transform(data['categorical_column'])

# 3. Feature Selection (SelectKBest with ANOVA F-statistic in this example)
X = data.drop('target_column', axis=1)  # Adjust the target_column name
y = data['target_column']
selector = SelectKBest(f_classif, k=5)  # Choose the number of top features to select
X_new = selector.fit_transform(X, y)

# 4. Outlier Detection and Removal (Optional)
# Use outlier detection methods like IQR, Z-score, or Isolation Forest to remove outliers.

# 5. Data Scaling (Standardization)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_new)

# 6. Choosing the Right Clustering Algorithm (K-means in this example)
kmeans = KMeans(n_clusters=3, random_state=0)
cluster_labels = kmeans.fit_predict(X_scaled)

# 7. Interpretation
# Interpret the clusters and gain insights from them. Print the cluster labels.
data['Cluster'] = cluster_labels
print(data.head())

# 8. Data Visualization (if needed)
# Visualize the clusters or selected features as needed.
# Example for 2D data visualization:
sns.scatterplot(x=data['Feature1'], y=data['Feature2'], hue=data['Cluster'], palette='Set1')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Clustered Data')
plt.show()
