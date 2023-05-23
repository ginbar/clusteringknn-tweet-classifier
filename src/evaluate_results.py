import argparse
from sklearn.metrics import recall_score, precision_score, f1_score
from infra.results import read_results, read_groundtruth
    

argument_parser = argparse.ArgumentParser("Evaluate model")
argument_parser.add_argument("hashtag", help="Hashtag used to create the dataset", type=str)
argument_parser.add_argument("--percentage", help="Percentage of the data to be used for training.", type=float, default=0.8)

args = argument_parser.parse_args()

metrics = [recall_score, precision_score, f1_score]

predicted = read_results(args.hashtag).astype(int)
ground_truth = read_groundtruth(args.hashtag)

for metric in metrics:
    print('Metric: ' + metric.__name__)
    print('score: ' + str(metric(ground_truth, predicted, average='macro')))
