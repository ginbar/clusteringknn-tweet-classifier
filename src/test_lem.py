import nltk
import pickle
import os

from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from sklearn.feature_extraction import DictVectorizer
from collections import Counter, OrderedDict

from sklearn.cluster import KMeans

# downloading nltk corpuses files
#nltk.download('stopwords')

lemmatizer = WordNetLemmatizer()
vectorizer = DictVectorizer()
model = KMeans()
model_file_path = 'data/models/test_model'

if not os.path.exists('data/models'):
    os.makedirs('data/models')

texts = [
    "the plural of feet is foot and the plural of dog is dogs",
    "the more you know the better you do",
    "the fast you go the least you enjoy",
    "for every wonderfull thing in this world there are plenty other equally worderfull things"
    "they've taken the stuff and put them on things",
    "the plural of tree is trees and the plural of cat is cats",
    "if lemmatize a word does this word became lemmatized or does it get lemmatized",
    "they've taken the stuff and put them on things",
    "they've taken the stuff and put them on things",
]

tokenized = [word_tokenize(txt) for txt in texts]
lemmatized = [[lemmatizer.lemmatize(t) for t in toknd] for toknd in tokenized]
data = vectorizer.fit_transform(Counter(d) for d in lemmatized)

model.fit(data)
model_bin = pickle.dumps(model)

print(texts)
print(tokenized)
print(lemmatized)
print(data)
print(data.A)
print(vectorizer.vocabulary_)
print(model.labels_)
print(model_bin)


with open(model_file_path, 'ab+') as f:
    byte_arr = bytes(model_bin)
    f.write(byte_arr)

