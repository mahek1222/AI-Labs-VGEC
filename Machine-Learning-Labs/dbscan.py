import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler



df=pd.read_csv(r"C:/Users/DELL/Downloads/customer.csv")

df

df.info()

df.rename(columns={'Annual Income (k$)':'Income',"Spending Score (1-100)":'SpendScore'},inplace=True)

df['gender'] = df['gender'].map({'Male': 0, 'Female': 1})
df=df.drop(['age'],axis=1)

df['gender'].value_counts()
x = df.iloc[:, [0,1,2]].values
from sklearn.cluster import DBSCAN
x = pd.get_dummies(x, drop_first=True)
scaler = StandardScaler()
x_scaled = scaler.fit_transform(x)

db=DBSCAN(eps=0,min_samples=5)
model = db.fit(x_scaled)

labels = model.labels_

labels

from sklearn import metrics

sam_core = np.zeros_like(labels,dtype=bool)

sam_core[db.core_sample_indices_] =True

len(db.core_sample_indices_)

len(set(labels))

n_clusters = len(set(labels))- (1 if -1 in labels else 0)

n_clusters