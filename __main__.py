import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.datasets import load_breast_cancer

cancer = load_breast_cancer()

df = pd.DataFrame(cancer['data'], columns=cancer['feature_names'])
df.head()
df.info() # n = 569, p = 30

# PCA Visualization
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

scaler = StandardScaler()
scaler.fit(df)

scaled_data = scaler.transform(df)

pca = PCA(n_components=2)
pca.fit(scaled_data)

X_pca = pca.transform(scaled_data)

scaled_data.shape # n = 569, p = 30
X_pca.shape # n = 569, p = 2

# Plot
def plot_pca():
    plt.figure(figsize=(8, 6))
    feat1 = X_pca[:,0]
    feat2 = X_pca[:,1]

    plt.scatter(x=feat1, y=feat2, c=cancer['target'], cmap='plasma', edgecolor='black')
    plt.xlabel('First Principal Component')
    plt.ylabel('Second Principal Component')

plot_pca()

pca.components_

df_component = pd.DataFrame(pca.components_, columns=cancer['feature_names'])

def heatmap():
    plt.figure(figsize=(12, 6))
    sns.heatmap(df_component, cmap='plasma')

heatmap()
