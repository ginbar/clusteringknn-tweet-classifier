import re
import stanza



hashtag_rgx = re.compile('\#[\S]+', re.VERBOSE | re.IGNORECASE) 
mentions_rgx = re.compile('@[\S]+', re.VERBOSE | re.IGNORECASE) 
rt_rgx = re.compile('RT\s@[\S]+:', re.VERBOSE | re.IGNORECASE)
url_rgx = re.compile('https?:\/\/.\S+', re.VERBOSE | re.IGNORECASE)



nlp = stanza.Pipeline('pt')

stopwords = set([sw[:-1] for sw in open('data/stopwords/pt-br.txt', 'r').readlines()])



def lemmatize(text):
    
    stripped = rt_rgx.sub('', text)
    stripped = hashtag_rgx.sub('', stripped)
    stripped = mentions_rgx.sub('', stripped)
    stripped = url_rgx.sub('', stripped)
    
    stripped = stripped.replace('\n', ' ')

    sentences = nlp(stripped).sentences
    lemmatized = [w.lemma for s in sentences for w in s.words if w.lemma not in stopwords]
    
    if len(lemmatized) == 0:
        return None
    
    return ' '.join(lemmatized) + '\n' 

