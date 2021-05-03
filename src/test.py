import math
import numpy as np
from models import ClusterTreeKNN, ClusteringPreprocessor
from sklearn.datasets import make_blobs
from numpy.random import randint

dataset_size = 500
dataset_dimensionality = 3
min_value = 0
max_value = 10
n_clusters = 20
n_sentiment_labels = 3 # positiv/neutral/negativ

# Fake clutering
dataset, cluster_labels = make_blobs(n_samples=dataset_size,
    n_features=dataset_dimensionality,
    centers=n_clusters,
    center_box=(min_value, max_value))

test_sample = randint(min_value, high=max_value + 1, size=dataset_dimensionality)

#TODO Refactor this class
preprocessor = ClusteringPreprocessor(n_clusters)
model = ClusterTreeKNN()

# clusters = preprocessor.create_clusters(dataset, n_clusters, cluster_labels)
clusters_masks = preprocessor.create_clusters_masks(dataset)

# TODO Evaluate clusters. Word cloud UI goes here

# Fake evaluation
# clusters = clusters[:n_clusters - 4] 
sentiment_labels = randint(0, high=n_sentiment_labels, size=len(n_clusters))

model.fit(dataset, clusters_masks, sentiment_labels)

model.predict(test_sample)
