import argparse
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import recall_score, precision_score, f1_score, roc_auc_score, roc_curve
from model.cluster_tree_knn import ClusterTreeKNN
from infra.results import read_results, read_ground_truth
    

argument_parser = argparse.ArgumentParser("Evaluate model")
argument_parser.add_argument("hashtag", help="Hashtag used to create the dataset", type=str)
argument_parser.add_argument("--percentage", help="Percentage of the data to be used for training.", type=float, default=0.8)

args = argument_parser.parse_args()

metrics = [recall_score, precision_score, f1_score, roc_auc_score, roc_curve]

predicted = read_results(args.hashtag)
ground_truth = read_ground_truth(args.hashtag)

for metric in metrics:
    print('Metric: ' + metric.__name__)
    print('score: ' + str(metric(ground_truth, predicted)))
