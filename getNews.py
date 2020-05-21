from newsapi import NewsApiClient
import json
# Init
newsapi = NewsApiClient(api_key='e1529d21c8c34fc49978ebc4e673ccf7')

uid = 0
all_articles = []

# Get articles for month of April----------------------------------#
day = 17
base_date1 = '2020-04-'
base_date2 = '2020-04-0'
while(day<=30):
    if(day<10):
        if(day==9):
            from_date = base_date2 + str(day)
            to_date = base_date1 + str(day+1)
        else:
            from_date = base_date2 + str(day)
            to_date = base_date2 + str(day+1)
    else:
        from_date = base_date1 + str(day)
        to_date = base_date1 + str(day+1)
    for i in range(1, 2):
        all_articles.append(newsapi.get_everything(q=None,
                                                   sources='bbc-news,financial-post,entertainment-weekly,cnn,espn,google-news-in',
                                                   from_param=from_date,
                                                   to=to_date,
                                                   language='en',
                                                   sort_by='publishedAt',
                                                   page=i,
                                                   page_size=100))
    day = day + 2


# Get articles for month of May------------------------#
base_date1 = '2020-05-'
base_date2 = '2020-05-0'

for day in range(1, 17, 2):
    if day < 10:
        if day == 9:
            from_date = base_date2 + str(day)
            to_date = base_date1 + str(day + 1)
        else:
            from_date = base_date2 + str(day)
            to_date = base_date2 + str(day + 1)
    else:
        from_date = base_date1 + str(day)
        to_date = base_date1 + str(day + 1)
    for i in range(1, 2):
        all_articles.append(newsapi.get_everything(q=None,
                                                   sources='bbc-news,financial-post,entertainment-weekly,cnn,espn,google-news-in',
                                                   from_param=from_date,
                                                   to=to_date,
                                                   language='en',
                                                   sort_by='publishedAt',
                                                   page=i,
                                                   page_size=100))


# push articles into final array-----------------------#
for z in all_articles:

    for article in z['articles']:
        article['id'] = uid
        uid = uid+1

final_arr = []

for x in all_articles:
    final_arr.extend(x['articles'])

print("TOTAL NEWS: ", len(final_arr))

with open('dataset/news_data.json', 'w') as outfile:
    json.dump(final_arr, outfile)