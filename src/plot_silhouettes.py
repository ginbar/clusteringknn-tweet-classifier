from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
from infra.dataset_reader import DatasetReader

from preprocessing.text_transforms import TextTransforms


transforms = TextTransforms(language='portuguese')

reader = DatasetReader('PL2630', 'train')
data = transforms.vectorize(reader.get_lemmatized_tweets())

min_k, max_k = 80, 100
range_n_clusters = [k for k in range(min_k, max_k + 1)]


for n_clusters in range_n_clusters:

    fig = plt.figure()
    fig.set_size_inches(18, 7)
    
    ax = fig.gca()

    ax.set_xlim([-0.1, 1])
    ax.set_ylim([0, len(data) + (n_clusters + 1) * 10])

    clusterer = KMeans(n_clusters=n_clusters, n_init=10, random_state=12)
    cluster_labels = clusterer.fit_predict(data)

    silhouette_avg = silhouette_score(data, cluster_labels)
    sample_silhouette_values = silhouette_samples(data, cluster_labels)

    y_lower = 10

    for i in range(n_clusters):
        ith_cluster_silhouette_values = sample_silhouette_values[cluster_labels == i]

        ith_cluster_silhouette_values.sort()

        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i

        color = cm.nipy_spectral(float(i) / n_clusters)
        ax.fill_betweenx(
            np.arange(y_lower, y_upper),
            0,
            ith_cluster_silhouette_values,
            facecolor=color,
            edgecolor=color,
            alpha=0.7,
        )

        ax.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))
        y_lower = y_upper + 10  
    
    ax.set_title(f"Análise de silhueta para o KMeans com k = {n_clusters}")
    ax.set_xlabel("Coeficiente silhueta")
    ax.set_ylabel("Agrupamento")

    ax.axvline(x=silhouette_avg, color="red", linestyle="--")

    ax.set_yticks([])
    ax.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])
    colors = cm.nipy_spectral(cluster_labels.astype(float) / n_clusters)
    

plt.show()