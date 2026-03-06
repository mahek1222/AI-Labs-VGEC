import pandas as pd
import numpy as np

data = pd.read_csv(r"C:\Users\DELL\Downloads\climate_data.csv")
data.head()

from sklearn.preprocessing import StandardScaler
# Normalize the features
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data)
# Display scaled data
print(pd.DataFrame(data_scaled, columns=data.columns).head())

from sklearn.manifold import TSNE
# Apply t-SNE to reduce to 2 dimensions
tsne = TSNE(
    n_components=2,
    perplexity=30,
    max_iter=300,   
    random_state=42
)
data_tsne = tsne.fit_transform(data_scaled)
# Convert the t-SNE results into a DataFrame for easy plotting
data_tsne_df = pd.DataFrame(data_tsne, columns=['t-SNE-1', 't-SNE-2'])

import seaborn as sns
import matplotlib.pyplot as plt
# Plot t-SNE results
plt.figure(figsize=(10, 8))
sns.scatterplot(x='t-SNE-1', y='t-SNE-2', data=data_tsne_df, palette='coolwarm', s=70, edgecolor='k', alpha=0.7)
plt.title('t-SNE Visualization of Climate Patterns')
plt.xlabel('t-SNE Dimension 1')
plt.ylabel('t-SNE Dimension 2')
plt.show()