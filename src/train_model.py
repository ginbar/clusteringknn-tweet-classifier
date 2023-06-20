import argparse
import numpy as np
from chronometer import Chronometer
from model.cluster_tree_knn import ClusterTreeKNN
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
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


cleaned_train_data, cleaned_preprocessing = remove_invalid_clusters(train_vectorized_data, preprocessing)
y = create_labeling(cleaned_preprocessing)

# model = ClusterTreeKNN(cleaned_preprocessing.centroids, cleaned_preprocessing.clustering_mask)

# params = {
#     'n_neighbors': [3, 5, 9, 11], 
#     'sigma_nearest_nodes': [2, 3, 4, 5],
#     'initial_hyperlevel_threshold': [5, 7, 9]
# }

# clf = GridSearchCV(
#     estimator=model,
#     param_grid=params,
#     n_jobs=-1,
#     verbose=3
# )

# clf.fit(train_vectorized_data, y)

# print(clf.best_params_)

# best_params = clf.best_params_

# model = ClusterTreeKNN(
#     cleaned_preprocessing.clustering_mask,
#     cleaned_preprocessing.centroids,
#     initial_hyperlevel_threshold=best_params['initial_hyperlevel_threshold'],
#     sigma_nearest_nodes=best_params['sigma_nearest_nodes'],
#     n_neighbors=best_params['n_neighbors']
# )

# model = ClusterTreeKNN(
#     cleaned_preprocessing.clustering_mask,
#     cleaned_preprocessing.centroids,
#     initial_hyperlevel_threshold=5,
#     sigma_nearest_nodes=5,
#     n_neighbors=7
# )

model = ClusterTreeKNN(
    cleaned_preprocessing.clustering_mask,
    cleaned_preprocessing.centroids,
    initial_hyperlevel_threshold=7,
    sigma_nearest_nodes=5,
    n_neighbors=7
)

model.fit(
    cleaned_train_data, 
    y
)

with Chronometer() as exec_time:
    results = model.predict(test_vectorized_data)

print(f'Exec time: {float(exec_time)}s')

save_results(args.hashtag, results)
