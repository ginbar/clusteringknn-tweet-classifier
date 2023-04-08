from ui.word_cloud_ui import WordCloudUI
from dtos.bottom_level_cluster import BottomLevelCluster
from preprocessing.clustering import ClusteringPreprocessor

#TODO Load data
dataset = []

#TODO Apply elbow or silhouette methods 
n_clusters = 4

preprocessor = ClusteringPreprocessor(n_clusters)

preprocessor.fit(dataset)

clusters = preprocessor.create_clusters()

gui = WordCloudUI(clusters=clusters)

gui.show()
