import argparse
import numpy as np
from sklearn.metrics import recall_score, precision_score, f1_score
from sklearn.neighbors import KNeighborsClassifier
from infra.dataset_reader import DatasetReader
from infra.preprocessing import read_preprocessing_results
from infra.results import read_groundtruth
from preprocessing.text_transforms import TextTransforms
from preprocessing.utils import create_labeling, remove_invalid_clusters
    

argument_parser = argparse.ArgumentParser("Evaluate KNN")
argument_parser.add_argument("hashtag", help="Hashtag used to create the dataset", type=str)
argument_parser.add_argument("--language", help="Data language.", type=str, default='portuguese')


args = argument_parser.parse_args()

metrics = [recall_score, precision_score, f1_score]

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

model = KNeighborsClassifier(n_neighbors=7)
model.fit(cleaned_train_data, y)

predicted = model.predict(test_vectorized_data)
groundtruth = read_groundtruth(args.hashtag)

for metric in metrics:
    print('Metric: ' + metric.__name__)
    print('score: ' + str(metric(groundtruth, predicted, average='macro')))

print('#')

labels = ['Neg', 'Neu', 'Pos']

for index, label in enumerate(labels):
    print(f'{label}: {(predicted==index).sum()}')

print('#')

for index, label in enumerate(labels):
    print(f'{label}: {(groundtruth==index).sum()}')

print('#')

for index, label in enumerate(labels):
    print(f'{label}: {(y==index).sum()}')