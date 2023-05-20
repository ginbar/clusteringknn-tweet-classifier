import matplotlib.pyplot as plt
import numpy as np
from sklearn import metrics
import argparse
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
from infra.results import read_results, read_groundtruth


argument_parser = argparse.ArgumentParser("Evaluate model")
argument_parser.add_argument("hashtag", help="Hashtag used to create the dataset", type=str)
argument_parser.add_argument("--percentage", help="Percentage of the data to be used for training.", type=float, default=0.8)

args = argument_parser.parse_args()

# predicted = read_results(args.hashtag)
# groundtruth = read_groundtruth(args.hashtag)

predicted = [1,0,-1,1,-1,0,1,1,-1,0,0,1]
groundtruth = [1,1,0,1,-1,1,-1,1,-1,1,0,1]

labels = ['Negativo', 'Neutro', 'Positivo']


for label in labels:

    fpr, tpr, thresholds = metrics.roc_curve(y, pred)

    roc_auc = metrics.auc(fpr, tpr)
    display = metrics.RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc,
                                    estimator_name='example estimator')
    display.plot()

    plt.show()