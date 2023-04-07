from sklearn.cluster import KMeans
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize



class ClusteringPreprocessor(object):
    """docstring for Preprocessor."""

    def __init__(self, n_clusters):
        super(ClusteringPreprocessor, self).__init__()
        self._model = KMeans(n_clusters=n_clusters)

    def create_clusters_masks(self, data):
        tokenized_tweets = self.tokenizer(data)
        self._model.fit(tokenized_tweets)
        return self._model.labels_


    def tokenizer(tweets: list[str]) -> list[list[str]]:
        #TODO Implement multiple languages  
        #TODO Try to improve performance
        #TODO Apply lematization
        langsw = stopwords.words('portuguese')
        return np.array(w for w in word_tokenize(t.lower()) if w.isalpha() and not w in langsw for t in tweets)
