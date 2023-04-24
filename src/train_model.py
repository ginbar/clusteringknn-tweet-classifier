import argparse
from model.cluster_tree_knn import ClusterTreeKNN
from preprocessing.text_transforms import TextTransforms
from preprocessing.utils import remove_invalid_clusters
from infra.preprocessing import read_preprocessing_results
from infra.dataset_reader import DatasetReader


argument_parser = argparse.ArgumentParser("Train model")
argument_parser.add_argument("hashtag", help="Hashtag used to create the dataset", type=str)
argument_parser.add_argument("--language", help="Data language.", type=str, default='portuguese')

args = argument_parser.parse_args()

preprocessing = read_preprocessing_results(args.hashtag)
train_dataset = DatasetReader(args.hashtag, 'train')
transforms = TextTransforms(language=args.language)

model = ClusterTreeKNN()

train_lemmatized_data = train_dataset.get_lemmatized_tweets()
train_vectorized_data = transforms.vectorize(train_lemmatized_data)

cleaned_train_data, cleaned_preprocessing = remove_invalid_clusters(train_vectorized_data, preprocessing)

model.fit(train_vectorized_data, cleaned_preprocessing.clustering_mask, cleaned_preprocessing.centroids)

test_dataset = DatasetReader(args.hashtag, 'test')

test_lemmatized_data = train_dataset.get_lemmatized_tweets()
test_vectorized_data = transforms.vectorize(test_lemmatized_data)

results = model.predict(test_vectorized_data)
