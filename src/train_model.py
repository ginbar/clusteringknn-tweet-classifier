import argparse
import numpy as np
from model.cluster_tree_knn import ClusterTreeKNN
from preprocessing.text_transforms import TextTransforms
from preprocessing.utils import remove_invalid_clusters, create_labeling
from infra.preprocessing import read_preprocessing_results
from infra.dataset_reader import DatasetReader
from infra.results import save_results

argument_parser = argparse.ArgumentParser("Train model")
argument_parser.add_argument("hashtag", help="Hashtag used to create the dataset", type=str)
argument_parser.add_argument("--percentage", help="Percentage of the data to be used for training.", type=float, default=0.8)
argument_parser.add_argument("--language", help="Data language.", type=str, default='portuguese')

args = argument_parser.parse_args()

transforms = TextTransforms(language=args.language)

preprocessing = read_preprocessing_results(args.hashtag)

train_dataset = DatasetReader(args.hashtag, 'train')
test_dataset = DatasetReader(args.hashtag, 'test')

train_lemmatized_data = train_dataset.get_lemmatized_tweets()
test_lemmatized_data = test_dataset.get_lemmatized_tweets()

vectorized_data = transforms.vectorize(np.concatenate([train_lemmatized_data, test_lemmatized_data]))

train_vectorized_data = vectorized_data[:len(train_lemmatized_data)]
test_vectorized_data = vectorized_data[-len(test_lemmatized_data):]

model = ClusterTreeKNN(initial_hyperlevel_threshold=5)

cleaned_train_data, cleaned_preprocessing = remove_invalid_clusters(train_vectorized_data, preprocessing)
y = create_labeling(preprocessing)

model.fit(
    train_vectorized_data, 
    y,
    cleaned_preprocessing.clustering_mask, 
    cleaned_preprocessing.centroids
)

results = model.predict(test_vectorized_data)

save_results(args.hashtag, results)
