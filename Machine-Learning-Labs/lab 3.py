import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

df = pd.read_csv(r"C:\Users\Dell\download\energy_data.csv")
df.head()

df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
df = df.dropna() # Drop missing values
df = df.set_index('timestamp')

X = df[['energy_consumption' , 'temperature' ]]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42)
    kmeans.fit(X_scaled)
    wcss.append(kmeans.inertia_)

# Plotting the Elbow Method
plt.figure(figsize=(10, 6))
plt.plot(range(1, 11), wcss, marker='o', linestyle='-')
plt.title('Elbow Method')
plt.xlabel('Number of Clusters')
plt.ylabel('WCSS')
plt. show()

best_k = 3
kmeans = KMeans(n_clusters=best_k, init='k-means++', random_state=42)
df['cluster'] = kmeans.fit_predict(X_scaled)

from sklearn.metrics import silhouette_score
silhouette_avg = silhouette_score(X_scaled, df['cluster'])
print(f'Silhouette Score: {silhouette_avg:.2f}')

plt.figure(figsize=(10, 6))
sns.scatterplot(x='energy_consumption', y='temperature', hue='cluster' , data=df, palette='viridis', s=60)
plt.title('Clusters of Energy Consumption Patterns')
plt.xlabel('Energy Consumption')
plt.ylabel('Temperature')
plt.legend(title='Cluster')
plt.show()