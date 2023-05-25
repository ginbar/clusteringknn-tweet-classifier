import argparse
import numpy as np
from infra.dataset_reader import DatasetReader
from ui.word_cloud_ui import WordCloudUI
from preprocessing.clustering_preprocessor import ClusteringPreprocessor
from preprocessing.text_transforms import TextTransforms
from preprocessing.utils import silhouette_scores
from dtos.preprocessing_results import PreprocessingResults
from infra.preprocessing import save_preprocessing_results


argument_parser = argparse.ArgumentParser("Rotulate dataset")
argument_parser.add_argument("hashtag", help="Hashtag used to create the dataset", type=str)
argument_parser.add_argument("--k", help="Number of clusters.", type=int, default=19)
argument_parser.add_argument("--language", help="Data language.", type=str, default='portuguese')

args = argument_parser.parse_args()

transforms = TextTransforms(language=args.language)

train_dataset = DatasetReader(args.hashtag, 'train')
test_dataset = DatasetReader(args.hashtag, 'test')

train_lemmatized_data = train_dataset.get_lemmatized_tweets()
test_lemmatized_data = test_dataset.get_lemmatized_tweets()

vectorized_data = transforms.vectorize(np.concatenate([train_lemmatized_data, test_lemmatized_data]))

train_vectorized_data = vectorized_data[:len(train_lemmatized_data)]

preprocessor = ClusteringPreprocessor(args.k, transforms)
preprocessor.fit(train_vectorized_data, train_lemmatized_data)

clusters = preprocessor.create_clusters()

gui = WordCloudUI(clusters=clusters)

gui.show()

centroids = [cluster.centroid for cluster in clusters]
clustering_mask = preprocessor.get_clustering_mask()
assigned_labels = gui.get_assigned_labels()

preprocessing_results = PreprocessingResults(centroids, clustering_mask, assigned_labels)

save_preprocessing_results(args.hashtag, preprocessing_results)