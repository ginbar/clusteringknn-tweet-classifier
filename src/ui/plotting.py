from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import matplotlib.style as style


def plot_sillhouete_values(range_n_clusters):

    avg_distance=[]
    
    for n_clusters in range_n_clusters:
        clusterer = KMeans(n_clusters=n_clusters, random_state=42).fit(X)
        avg_distance.append(clusterer.inertia_)

        style.use("fivethirtyeight")
        plt.plot(range_n_clusters, avg_distance)
        plt.xlabel("Number of Clusters (k)")
        plt.ylabel("Distance")
        plt.show()