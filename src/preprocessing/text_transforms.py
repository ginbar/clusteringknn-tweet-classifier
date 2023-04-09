import re
import stanza
import numpy as np
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction import DictVectorizer
from scipy.sparse import csr_matrix
from collections import Counter

########### Downloads NLTK Corpuses

import nltk
#TODO Improve this method
try:
    nltk.find('stopwords')
except LookupError:
    nltk.download('stopwords')
    nltk.download('punkt')


from nltk.corpus import stopwords

###################################


hashtag_rgx = re.compile('\#[\S]+', re.VERBOSE | re.IGNORECASE) 
mentions_rgx = re.compile('@[\S]+', re.VERBOSE | re.IGNORECASE) 
rt_rgx = re.compile('RT\s@[\S]+:', re.VERBOSE | re.IGNORECASE)
url_rgx = re.compile('https?:\/\/.\S+', re.VERBOSE | re.IGNORECASE)




class TextTransforms(object):
    """
    Generates the bottom layer clusters to be labeled.
    
    Parameters
    ----------
    n_clusters : int
        Number of clusters to be generated.

    language : str
        Language of the data to be inputed.
    """
    
    def __init__(self, language:str='portuguese'):
        self._language = language
        self._nlp = stanza.Pipeline('pt')

    
    
    def lemmatize(self, text:str) -> str:
        """
        Lemmatizes a given text string using Stanza's Pipelines.

        Parameters:
        - text (str): A string of text to be lemmatized.

        Returns:
        - str: A string of text that has been lemmatized.
        """
        
        stripped = rt_rgx.sub('', text)
        stripped = hashtag_rgx.sub('', stripped)
        stripped = mentions_rgx.sub('', stripped)
        stripped = url_rgx.sub('', stripped)
        
        stripped = stripped.replace('\n', ' ')

        sentences = nlp(stripped).sentences
        lemmatized = [w.lemma for s in sentences for w in s.words if w.lemma not in stopwords.words(self._language)]
        
        if len(lemmatized) == 0:
            return None
        
        return ' '.join(lemmatized) + '\n' 



    def vectorize(self, texts: list[str]) -> np.ndarray:
        """
        Vectorize a list of texts.

        Parameters:
        - texts (str): A string of text to be lemmatized.

        Returns:
        - str: A string of text that has been lemmatized.
        """

        #TODO Try to improve performance

        langsw = set(stopwords.words(self._language))

        tokenized_tweets = (Counter(w for w in word_tokenize(t.lower(), self._language) if w.isalpha() and not w in langsw) for t in texts)

        vectorizer =  DictVectorizer(sparse=True)
        print(tokenized_tweets)
        vect_data = vectorizer.fit_transform(tokenized_tweets)
        print(vect_data)

        # sums = np.sum(vect_data, axis=1).A[:, 0]
        # N = len(s)
        # divisor = csr_matrix((np.reciprocal(s), (range(N), range(N))))
        # vect_data = divisor * vect_data
        
        return vect_data
