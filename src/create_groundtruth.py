import argparse
import numpy as np
from infra.dataset_reader import DatasetReader
from infra.results import save_groundtruth
from ui.labelling_ui import LabellingUI


argument_parser = argparse.ArgumentParser("Rotulate dataset")
argument_parser.add_argument("hashtag", help="Hashtag used to create the dataset", type=str)
argument_parser.add_argument("--percentage", help="Percentage of the data to be used for training.", type=float, default=0.8)
argument_parser.add_argument("--language", help="Data language.", type=str, default='portuguese')

args = argument_parser.parse_args()

train_dataset = DatasetReader(args.hashtag, 'train')

raw_tweets = train_dataset.get_raw_tweets()

gui = LabellingUI(raw_tweets)

gui.show()

groundtruth = gui.get_assigned_labels()

print(groundtruth)

save_groundtruth(args.hashtag, groundtruth)
