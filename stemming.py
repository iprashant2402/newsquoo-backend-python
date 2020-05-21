import nltk
from nltk import pos_tag
from nltk.stem import PorterStemmer
from nltk import word_tokenize
import re
import pandas as pd

articles = pd.read_json("./dataset/news_data.json",orient="records")
stemmer = PorterStemmer()
tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
progress = 0 
def stem(x):
    dirty = word_tokenize(x)
    tokens = []
    for word in dirty:
        if word.strip('.') == '': 
           pass
        elif re.search(r'\d+', word):
           pass
        else:
           tokens.append(word.strip('.'))
    global start
    global progress
    tokens = pos_tag(tokens) 
    progress += 1
    stems = ' '.join(stemmer.stem(key.lower()) for key, value in  tokens if value != 'NNP') #getting rid of proper nouns
 
    # end = time.time()
    # sys.stdout.write('\r {} percent, {} position, {} per second '.format(str(float(progress / len(articles))), 
    #                 str(progress), (1 / (end - start))))  
    # start = time.time()
    return stems

print("Dataset before Stemming : ")
print(articles.info())

articles['content'].dropna(inplace=True)
articles['stems'] = articles['content'].apply(lambda x: stem(x))
#new_articles = articles.apply(lambda x: stem(x) if x.name == 'content' else x)
#print(new_articles.head(1))
print("Dataset after Stemming : ")
print(articles.info())
