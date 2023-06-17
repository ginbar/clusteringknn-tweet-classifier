import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import pyplot as plt
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import roc_curve, auc, RocCurveDisplay
from infra.results import read_results, read_groundtruth


argument_parser = argparse.ArgumentParser("Plot ROC curves")
argument_parser.add_argument("hashtag", help="Hashtag used to create the dataset", type=str)
argument_parser.add_argument("--percentage", help="Percentage of the data to be used for training.", type=float, default=0.8)

args = argument_parser.parse_args()

predicted = read_results(args.hashtag)
groundtruth = read_groundtruth(args.hashtag)

labels = ['Negativo', 'Neutro', 'Positivo']


def one_vs_all(array, value):
    pos_mask = array == value
    neg_mask = array != value
    
    new_array = array.copy()

    np.putmask(new_array, pos_mask, 1)
    np.putmask(new_array, neg_mask, 0)
    
    return new_array


for label, label_name in enumerate(labels):

    ovsa_groundtruth = one_vs_all(groundtruth, label)
    ovsa_predicted = one_vs_all(predicted, label)
    
    fpr, tpr, thresholds = roc_curve(ovsa_groundtruth, ovsa_predicted)

    roc_auc = auc(fpr, tpr)
    
    fig = plt.figure()
    fig.set_size_inches(18, 7)
    
    ax = fig.gca()

    ax.set_title(label_name)
    ax.set_xlabel("Taxa de falsos positivos")
    ax.set_ylabel("Taxa de verdadeiros positivos")

    display = RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc)

    display.plot(ax)

plt.show()
