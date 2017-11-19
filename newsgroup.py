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
