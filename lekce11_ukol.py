from sklearn.datasets import load_digits
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, silhouette_samples

#Silhouette:
#1-B
#2-C
#3-D
#4-A

#MNIST:
digits = load_digits()
X = digits.data
scaler = StandardScaler()
X = scaler.fit_transform(X)
#print(X.shape)

tsne = TSNE(
    init="pca",
    n_components=2,
    perplexity=10,
    learning_rate="auto",
    random_state=0,
)
X = tsne.fit_transform(X)
#print(X.shape)

plt.scatter(X[:, 0], X[:, 1], s=50) #odhaduji 10 clusterů
#plt.show()

model = KMeans(n_clusters=10, random_state=0)
labels = model.fit_predict(X)
plt.scatter(X[:, 0], X[:, 1], c=labels, s=50, cmap="Set1")
centers = model.cluster_centers_
plt.scatter(centers[:, 0], centers[:, 1], c="black", s=200, alpha=0.5)
plt.show()

print(silhouette_score(X, labels)) #výsledek pro 10 clusterů je 0.55, čím vyšší a blíže 1, tím lépe definované clustery