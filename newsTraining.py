import sys
import numpy as np
import pandas as pd
import pickle

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from sklearn.cluster import MiniBatchKMeans
import nltk
from nltk import pos_tag
from nltk.stem import PorterStemmer
from nltk import word_tokenize
import re
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
import json


X = pd.read_json("./dataset/news_data.json",orient="records")
X = X[pd.isna(X['title'])==False]
X = X[pd.isna(X['content'])==False]

stemmer = PorterStemmer()
tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
progress = 0 
def stem(x):
    dirty = word_tokenize(x)
    tokens = []
    for word in dirty:
        if word.strip('.') == '': 
           pass
        elif re.search(r'\d{1,}', word): 
           pass
        else:
           tokens.append(word.strip('.'))
    global start
    global progress
    tokens = pos_tag(tokens) #
    progress += 1
    stems = ' '.join(stemmer.stem(key.lower()) for key, value in  tokens if value != 'NNP') #getting rid of proper nouns
 
    # end = time.time()
    # sys.stdout.write('\r {} percent, {} position, {} per second '.format(str(float(progress / len(articles))), 
    #                 str(progress), (1 / (end - start)))) #lets us see how much time is left 
    # start = time.time()
    return stems

X['content'].dropna(inplace=True)
X['stems'] = X['content'].apply(lambda x: stem(x))


print(X.info())
text_content = X['stems']
vector = TfidfVectorizer(max_df=0.3,         
                         min_df=8,
                         stop_words='english',
                         lowercase=True,
                         use_idf=True,
                         norm=u'l2',
                         smooth_idf=True
                            )

tfidf = vector.fit_transform(text_content)

pickle.dump(X, open('output/X', 'wb')) 
pickle.dump(vector, open('output/vector', 'wb')) 
pickle.dump(tfidf, open('output/tfidf', 'wb'))

updated_news_data = X.to_json("dataset/filtered_news_data.json", orient="records")