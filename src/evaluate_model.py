import argparse
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from model.cluster_tree_knn import ClusterTreeKNN
from infra.dataset_reader import DatasetReader


argument_parser = argparse.ArgumentParser("Evaluate model")
argument_parser.add_argument("hashtag", help="Hashtag used to create the dataset", type=str)
argument_parser.add_argument("--percentage", help="Percentage of the data to be used for training.", type=float, default=0.8)

args = argument_parser.parse_args()

train_dataset = DatasetReader(args.hashtag, 'train')
test_dataset = DatasetReader(args.hashtag, 'test')

model = ClusterTreeKNN(initial_hyperlevel_threshold=5)

params = {
    'n_neighbors': [3,5,7,9,11,13], 
    'sigma_nearest_nodes': [3,5,7,9,11,13],
    'initial_hyperlevel_threshold': [3,5,7,9,11,13],
}

clf = GridSearchCV(
    estimator=model,
    param_grid=params,
    cv=5,
    n_jobs=5,
    verbose=3
)

print(clf.best_params_)
