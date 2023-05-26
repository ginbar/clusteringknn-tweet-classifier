import argparse
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
from infra.results import read_results, read_groundtruth


argument_parser = argparse.ArgumentParser("Evaluate model")
argument_parser.add_argument("hashtag", help="Hashtag used to create the dataset", type=str)
argument_parser.add_argument("--percentage", help="Percentage of the data to be used for training.", type=float, default=0.8)

args = argument_parser.parse_args()

predicted = read_results(args.hashtag)
groundtruth = read_groundtruth(args.hashtag)

cm = confusion_matrix(groundtruth, predicted)

cm_display = ConfusionMatrixDisplay(cm).plot()

plt.xlabel("Resultado")
plt.ylabel("Groundtruth")

plt.show()