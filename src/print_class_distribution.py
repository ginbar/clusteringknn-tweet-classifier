import argparse
from sklearn.metrics import recall_score, precision_score, f1_score
from infra.results import read_results, read_groundtruth
    

argument_parser = argparse.ArgumentParser("Print class distribution")
argument_parser.add_argument("hashtag", help="Hashtag used to create the dataset", type=str)

args = argument_parser.parse_args()

predicted = read_results(args.hashtag).astype(int)
groundtruth = read_groundtruth(args.hashtag)

arrays = [predicted, groundtruth]
labels = ['Neg', 'Neu', 'Pos']

for arr in arrays:
    print('#')
    for index, label in enumerate(labels):
        print(f'{label}: {(arr == index).sum()}')
        