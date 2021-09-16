import nltk
import re

from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords


# downloading nltk corpuses files
# nltk.download('stopwords')

hashtag_rgx = re.compile('\#[A-Za-z0-9]+', re.VERBOSE | re.IGNORECASE) 
mentions_rgx = re.compile('@[A-Za-z0-9]+', re.VERBOSE | re.IGNORECASE) 
rt_rgx = re.compile('RT [\s]+:', re.VERBOSE | re.IGNORECASE)
url_rgx = re.compile('https?:\/\/.\S+', re.VERBOSE | re.IGNORECASE)


lemmatizer = WordNetLemmatizer()


stopwords = set([sw[:-1] for sw in open('data/stopwords/pt-br.txt', 'r').readlines()])


def lemmatize(text):
    
    stripped = hashtag_rgx.sub('', text)
    stripped = mentions_rgx.sub('', stripped)
    stripped = rt_rgx.sub('', stripped)
    stripped = url_rgx.sub('', stripped)
    
    stripped = stripped.replace('\n', ' ')

    tokens = word_tokenize(stripped)
    lemmatized = [lemmatizer.lemmatize(t) for t in tokens if t not in stopwords]

    if len(lemmatized) == 0:
        return None
    
    return ' '.join(lemmatized) + '\n' 

