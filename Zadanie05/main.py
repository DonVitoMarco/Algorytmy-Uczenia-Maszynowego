import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn import metrics
from sklearn.preprocessing import LabelEncoder

# Load data
path = "./data/iris.data"
names = ["sepal_length", "sepal_width", "petal_length", "petal_width", "class"]
dataset = pd.read_csv(path, names=names)

# Split dataset
X = dataset.iloc[:, :4]
y = dataset['class']

# Normalize class
encoder = LabelEncoder()
y = encoder.fit_transform(y)

# Show ground truth
# plt.scatter(X.iloc[:, 0], X.iloc[:, 1], c=y)
# plt.xlabel(names[0])
# plt.ylabel(names[1])
# plt.show()

# K-means for first two attributes
km = KMeans(n_clusters=3)
y_km_prediction = km.fit_predict(X.iloc[:, :2])
centroids = km.cluster_centers_

score_ari_2attributes = metrics.adjusted_rand_score(y, y_km_prediction)

plt.scatter(X.iloc[:, 0], X.iloc[:, 1], c=y_km_prediction)
plt.scatter(centroids[:, 0], centroids[:, 1], marker='x', s=180, linewidths=3, color='b')
plt.xlabel(names[0])
plt.ylabel(names[1])
plt.show()

# K-means for all attributes
km = KMeans(n_clusters=3)
y_km_prediction_4attributes = km.fit_predict(X)
score_ari_4attributes = metrics.adjusted_rand_score(y, y_km_prediction_4attributes)

# Print result
print("K-means ARI (Adjusted Rand Index) for 2 attributes: ", score_ari_2attributes)
print("K-means ARI (Adjusted Rand Index) for all attributes: ", score_ari_4attributes)
