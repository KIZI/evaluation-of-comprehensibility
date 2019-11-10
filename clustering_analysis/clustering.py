import re
from collections import OrderedDict
from nltk import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
import mysql.connector

# Data from DB

cnx = mysql.connector.connect(host='', user='', password='',
                                  database='')
cursor = cnx.cursor()

troll_sql = """
        select author, content, account_category from russian_troll_tweets_author_sample_1 order by author
"""

print('Executing SQL')
#
cursor.execute(troll_sql)
print('SQL executed')


documents_dict = OrderedDict()
true_labels = []
for (author, content, account_category) in cursor:
    if author in documents_dict:
        documents_dict[author] += '\n' + content
    else:
        true_labels.append(account_category)
        documents_dict[author] = content

# Document-term matrix

class TrollTfidfVectorizer(TfidfVectorizer):

    def __init__(self, *args, **kwargs):
        troll_stop_words = {'don', 'just', 'like'}
        kwargs['stop_words'] = set(ENGLISH_STOP_WORDS).union(troll_stop_words)
        kwargs['preprocessor'] = self.vectorizer_preprocess
        self.wnl = WordNetLemmatizer()
        super(TrollTfidfVectorizer, self).__init__(*args, **kwargs)

    def build_analyzer(self):
        analyzer = super(TfidfVectorizer, self).build_analyzer()
        return lambda doc: ([self.wnl.lemmatize(w) for w in analyzer(doc)])

    def vectorizer_preprocess(self, s):
        # remove urls
        s = re.sub(r'(https?|ftp)://(-\.)?([^\s/?\.#-]+\.?)+(/[^\s]*)?', '', s)
        # remove amp
        s = s.replace('&amp;', '')
        # remove RT signs (no meaning) but keep username
        s = re.sub(r'\bRT\b\s+', '', s)
        s = s.lower()
        return s


vectorizer = TrollTfidfVectorizer()
doc_term_matrix = vectorizer.fit_transform(documents_dict.values())

vectorizer = TrollTfidfVectorizer(min_df=0.4)
doc_term_matrix = vectorizer.fit_transform(documents_dict.values())

# Number of clusters

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as pl

model_scores = []
model_silhouettes = []
clusters_range = range(2, 30)

for n_clusters in clusters_range:
    print(f'Test clustering with {n_clusters}')

    model = KMeans(n_clusters=n_clusters, init='k-means++', n_init=1)
    cluster_labels = model.fit_predict(doc_term_matrix)

    score = model.score(doc_term_matrix)
    print(f'Model score is {model.score(doc_term_matrix)}')
    model_scores.append(score)

    silhouette_avg = silhouette_score(doc_term_matrix, cluster_labels)
    model_silhouettes.append(silhouette_avg)
    print(f'Avg silhouette score is {silhouette_avg}')

pl.plot(clusters_range, model_scores)
pl.xlabel('Number of Clusters')
pl.ylabel('Score')
pl.title('Elbow Curve')
pl.show()

pl.plot(clusters_range, model_silhouettes)
pl.xlabel('Number of Clusters')
pl.ylabel('Score')
pl.title('Average Silhouette score')
pl.show()

# Training

from sklearn.cluster import KMeans
model = KMeans(n_clusters=6, init='k-means++')
cluster_labels = model.fit_predict(doc_term_matrix)

from nltk.cluster import KMeansClusterer, cosine_distance
model = KMeansClusterer(6, distance=cosine_distance,
                        repeats=25, avoid_empty_clusters=True)
data = doc_term_matrix.toarray()
cluster_labels = model.cluster(data, assign_clusters=True)

from sklearn.cluster import AgglomerativeClustering
model = AgglomerativeClustering(n_clusters=6, affinity='euclidean',
                                linkage='single')
cluster_labels = model.fit_predict(data.toarray())

# Metrics

from sklearn.metrics import adjusted_rand_score,\
    silhouette_score, davies_bouldin_score,\
    calinski_harabaz_score, v_measure_score, fowlkes_mallows_score

silhouette_avg = silhouette_score(doc_term_matrix, cluster_labels)
db_index = davies_bouldin_score(doc_term_matrix, cluster_labels)
vrc = calinski_harabaz_score(doc_term_matrix, cluster_labels)
adjusted_rand = adjusted_rand_score(true_labels, cluster_labels)
v_measure = v_measure_score(true_labels, cluster_labels)
fm_score = fowlkes_mallows_score(true_labels, cluster_labels)

# Word clouds

from wordcloud import WordCloud
import matplotlib.pyplot as pl

def wordcloud(dataframe, spectral=False):
    clusters_word_freq = []

    for index, row in dataframe.iterrows():
        freq_dict = {}
        for col_name in dataframe.columns:
            if row[col_name] > 0.00001:
                freq_dict[col_name] = float(row[col_name])
        clusters_word_freq.append(freq_dict)

    fig = pl.figure(figsize=(20, 10))
    for cluster, freq_dict in enumerate(clusters_word_freq):
        if spectral:
            def color_func(word, *args, **kwargs):
                cmap = pl.cm.get_cmap('coolwarm')
                rgb = cmap(freq_dict[word] / 100, bytes=True)[0:3]
                return rgb
        else:
            color_func = None

        ax = fig.add_subplot(2, 3, cluster + 1)
        cloud = WordCloud(normalize_plurals=False,
                          background_color='white', color_func=color_func)
        cloud.generate_from_frequencies(frequencies=freq_dict)
        ax.imshow(cloud, interpolation='bilinear')
        ax.set_yticks([])
        ax.set_xticks([])
        ax.text(0.35, 1, f'Cluster {cluster}',
                 fontsize=32, va='bottom',transform=ax.transAxes)
    fig.show()

from pandas import DataFrame

dataframe = DataFrame(model._means,
                      columns=vectorizer.get_feature_names())
wordcloud(dataframe)

# Z-score charts

from scipy.stats import stats

zscore = stats.zscore(model._means, axis=0)
zscore_df = DataFrame(zscore, columns=vectorizer.get_feature_names())

import matplotlib.pyplot as pl
from pandas import concat

def zscore(dataframe, term_count=10):

    fig = pl.figure(figsize=(15, 20))
    for i in range(dataframe.shape[0]):
        cluster_df = dataframe.iloc[[i]].T
        cluster_df = cluster_df.rename(index=str, columns={i: 'Z-score'})
        cluster_df = cluster_df.sort_values(by=['Z-score'])
        if term_count:
            half_term_count = term_count // 2
            sliced_df = concat([cluster_df[:half_term_count],
                                cluster_df[-half_term_count:]])
        else:
            sliced_df = cluster_df

        high_df = sliced_df.copy()
        high_df.loc[high_df['Z-score'] < 0, 'Z-score'] = 0
        low_df = sliced_df.copy()
        low_df.loc[low_df['Z-score'] >= 0, 'Z-score'] = 0
        high_zscores = high_df['Z-score']
        low_zscores = low_df['Z-score']
        ind = sliced_df.index

        ax = fig.add_subplot(3, 2, i + 1)

        ax.barh(ind, low_zscores, label='Low')
        ax.barh(ind, high_zscores, label='High')
        ax.set_xlabel(f'Z-scores of Cluster {i}', fontsize=26)
        ax.set_ylabel('Term', fontsize=18)
        ax.tick_params(labelsize=18)
        ax.legend(prop={'size': 18})
        ax.grid(True)

    fig.subplots_adjust(hspace=0.8)
    fig.show()

zscore(zscore_df)

# Z-score word clouds

from sklearn.preprocessing import MinMaxScaler

zscore = stats.zscore(model._means, axis=0)

helper = zscore.reshape(zscore.shape[0] * zscore.shape[1], 1)
scaler = MinMaxScaler((0,100))
helper = scaler.fit_transform(helper)
zscore = helper.reshape(zscore.shape[0], zscore.shape[1])

zcore_df = DataFrame(zscore, columns=vectorizer.get_feature_names())

wordcloud(zcore_df, spectral=True)