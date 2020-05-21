import numpy as np
import pandas as pd
import pickle
import json

import matplotlib.pyplot as plt
import matplotlib.cm as cm

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

X = pickle.load(open('./output/X', 'rb'))
vector = pickle.load(open('./output/vector', 'rb'))
tfidf = pickle.load(open('./output/tfidf', 'rb'))

with open("./dataset/filtered_news_data.json") as f:
    newsData = json.load(f)

for i in range(0, len(newsData)):
    newsData[i]["index"] = i

# print(newsData)
# print(len(newsData))

def search(tfidf_matrix, model, query_request, top_n=10):
    request_transform = model.transform([query_request])
    similarity = np.dot(request_transform, np.transpose(tfidf_matrix))
    x = np.array(similarity.toarray()[0])
    print("LINE 23: ")
    print(x)
    indices=np.argsort(x)[-10:][::-1]
    return indices

def find_similar(tfidf_matrix, index, top_n = 5):
    print("---REQUESTED TITLE---")
    print(X['title'].iloc[index])
    cosine_similarities = linear_kernel(tfidf_matrix[index:index+1], tfidf_matrix).flatten()
    related_docs_indices = [i for i in cosine_similarities.argsort()[::-1] if i != index]
    return [index for index in related_docs_indices][0:top_n]    

def print_result(indices,X):
    print('\nBest Results :')
    for i in indices:
        print(i)
        print(X['title'].iloc[i])


def find_similar_news(i):
    news = []
    indices = find_similar(tfidf, i, top_n=10)
    for j in indices:
        news.append(newsData[j])
    return news

def initialize_news_stack():
    stack = []

    def sortStack(e):
        return e["publishedAt"]

    for i in range((len(newsData)-10), len(newsData)):
        source = newsData[i].get("index")
        title = newsData[i].get("title")
        #print("PUBLISHED AT: " + str(newsData[i].get("publishedAt")) + " ### " + str(title)+" ------------------------------> " + str(source))
        stack.append(newsData[i])

    return stack

#initialize_news_stack()

# request = 'india'
# request = text_content[0]

# result = search(tfidf, vector, request, top_n=10)
# print_result(request, result, X)

# index = 991
# result = find_similar(tfidf, index, top_n=3)
# print(result)
# print_result(result, X)


def find_optimal_clusters(data, max_k):
    iters = range(2, max_k+1, 2)
    
    sse = []
    for k in iters:
        sse.append(MiniBatchKMeans(n_clusters=k).fit(data).inertia_)
        print('Fit {} clusters'.format(k))
        
    f, ax = plt.subplots(1, 1)
    ax.plot(iters, sse, marker='o')
    ax.set_xlabel('Cluster Centers')
    ax.set_xticks(iters)
    ax.set_xticklabels(iters)
    ax.set_ylabel('SSE')
    ax.set_title('SSE by Cluster Center Plot')


#find_optimal_clusters(tfidf,20)
#clusters = MiniBatchKMeans(n_clusters=14).fit_predict(tfidf)


def get_top_keywords(data, clusters, labels, n_terms):
    df = pd.DataFrame(data.todense()).groupby(clusters).mean()
    
    for i, r in df.iterrows():
        print('\nCluster {}'.format(i))
        print(','.join([labels[t] for t in np.argsort(r)[-n_terms:]]))


#get_top_keywords(tfidf, clusters, vector.get_feature_names(), 10)