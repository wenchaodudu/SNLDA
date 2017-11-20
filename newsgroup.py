from sklearn.datasets import fetch_20newsgroups

newsgroups_train = fetch_20newsgroups(subset='train')
newsgroup_test = fetch_20newsgroups(subset='test')
train_target = newsgroups_train.target
from sklearn.feature_extraction.text import TfidfVectorizer

categories = ['alt.atheism', 'talk.religion.misc', 'comp.graphics', 'sci.space']
newsgroups_train = fetch_20newsgroups(subset='train', categories=categories)
vectorizer = TfidfVectorizer()
vectors = vectorizer.fit_transform(newsgroups_train.data)

# vectors: TF-IDF vectors, 2034 * 34118
# newsgroups_train.target: label for each sample
# newsgroups_train.target_names: label name

## LDA fitting
from sklearn.decomposition import LatentDirichletAllocation as LDA
color = {0: 'g', 1: 'b', 2: 'r', 3: 'y', 4: 'black'}
colors = [color[x] for x in clusters]
solution = LDA(n_components=20, learning_method='batch', max_iter=30).fit_transform(vectors)