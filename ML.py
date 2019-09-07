import pandas as pd
from tqdm import tqdm
tqdm.pandas()

import numpy as np

import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from bs4 import BeautifulSoup

df = pd.read_excel('trainingObamaRomneytweets.xlsx',sheet_name=1)

# df_test = pd.read_csv('test.csv')

df = df[df['Class'] != '!!!!' ]
df = df[df['Class'] != 'IR' ]
df = df[df['Class'] != 'irrelevant']
df = df[df['Class'] != 'irrevelant']

df = df[df['Class'] != '' ]
df = df[df['Class'].notnull()]
df = df[df['Class'] != 2]
df = df[df['Class'] != '2']

df['Anootated tweet'] = df['Anootated tweet'].apply(str)
# print(df.dtypes)
# print(df)


df['Class'] = df['Class'].astype(int)
df['Class'] = df['Class'].replace(-1, 2)


from collections import Counter
c = Counter(df['Class'].values)
print(c)

df['Anootated tweet'] = df['Anootated tweet'].apply(lambda x: BeautifulSoup(str(x), 'lxml').get_text())
# print(df['Anootated tweet'].values)
# print(aa)

df['Anootated tweet'] = df['Anootated tweet'].apply(lambda x: re.sub('https?://[A-Za-z0-9./]+', '', x))

df['Anootated tweet'] = df['Anootated tweet'].apply(lambda x: re.sub('http?://[A-Za-z0-9./]+', '', x))

# import unicodedata
# df['Anootated tweet'] = unicodedata.normalize('NFKD', df['Anootated tweet'].values)

# print("----- decontracted ------")

# df['Anootated tweet'] = df['Anootated tweet'].apply(lambda x: re.sub(r"(W|w)on(\'|\’)t", "will not", x))
# df['Anootated tweet'] = df['Anootated tweet'].apply(lambda x: re.sub(r"(C|c)an(\'|\’)t", "can not", x))
# df['Anootated tweet'] = df['Anootated tweet'].apply(lambda x: re.sub(r"(Y|y)(\'|\’)all", "you all", x))
# df['Anootated tweet'] = df['Anootated tweet'].apply(lambda x: re.sub(r"(Y|y)a(\'|\’)ll", "you all", x))
# df['Anootated tweet'] = df['Anootated tweet'].apply(lambda x: re.sub(r"(I|i)(\'|\’)m", "i am", x))
# df['Anootated tweet'] = df['Anootated tweet'].apply(lambda x: re.sub(r"(A|a)in(\'|\’)t", "aint", x))
# df['Anootated tweet'] = df['Anootated tweet'].apply(lambda x: re.sub(r"n(\'|\’)t", " not", x))
# df['Anootated tweet'] = df['Anootated tweet'].apply(lambda x: re.sub(r"(\'|\’)re", " are", x))
# df['Anootated tweet'] = df['Anootated tweet'].apply(lambda x: re.sub(r"(\'|\’)s", " is", x))
# df['Anootated tweet'] = df['Anootated tweet'].apply(lambda x: re.sub(r"(\'|\’)d", " would", x))
# df['Anootated tweet'] = df['Anootated tweet'].apply(lambda x: re.sub(r"(\'|\’)ll", " will", x))
# df['Anootated tweet'] = df['Anootated tweet'].apply(lambda x: re.sub(r"(\'|\’)t", " not", x))
# df['Anootated tweet'] = df['Anootated tweet'].apply(lambda x: re.sub(r"(\'|\’)ve", " have", x))


print('---correct spelling----')
mispell_dict = {'colour': 'color', 'centre': 'center', 'favourite': 'favorite', 'travelling': 'traveling', 'counselling': 'counseling', 'theatre': 'theater', 'cancelled': 'canceled', 'labour': 'labor', 'organisation': 'organization', 'wwii': 'world war 2', 'citicise': 'criticize', 'youtu ': 'youtube ', 'Qoura': 'Quora', 'sallary': 'salary', 'Whta': 'What', 'narcisist': 'narcissist', 'howdo': 'how do', 'whatare': 'what are', 'howcan': 'how can', 'howmuch': 'how much', 'howmany': 'how many', 'whydo': 'why do', 'doI': 'do I', 'theBest': 'the best', 'howdoes': 'how does', 'mastrubation': 'masturbation', 'mastrubate': 'masturbate', "mastrubating": 'masturbating', 'pennis': 'penis', 'Etherium': 'Ethereum', 'narcissit': 'narcissist', 'bigdata': 'big data', '2k17': '2017', '2k18': '2018', 'qouta': 'quota', 'exboyfriend': 'ex boyfriend', 'airhostess': 'air hostess', "whst": 'what', 'watsapp': 'whatsapp', 'demonitisation': 'demonetization', 'demonitization': 'demonetization', 'demonetisation': 'demonetization'}
def correct_spelling(x, dic):
    for word in dic.keys():
        x = x.replace(word, dic[word])
    return x

df['Anootated tweet'] = df['Anootated tweet'].progress_apply(lambda x: correct_spelling(x, mispell_dict))

# from nltk.stem import PorterStemmer
# ps = PorterStemmer()
# # print(short_data['Anootated tweet'].values)

# print('---------- Stemming ---------')
# df['Anootated tweet'] = df['Anootated tweet'].apply(lambda x: ' '.join( [ ps.stem(word) for word in x.split() ]))
# # print(df['Anootated tweet'].values)



from nltk.corpus import stopwords
stop = stopwords.words("english")
# print(df['Anootated tweet'].values.tolist())

print("----------- Remove Stop Word -------------")
df["Anootated tweet"] = df["Anootated tweet"].apply(lambda x: ' '.join( word for word in x.split() if word not in stop ))


print ("------ Lower Case -------")
df['Anootated tweet'] = df['Anootated tweet'].apply(lambda x: ' '.join(x.lower() for x in x.split()))

print ("------ remove punctuation ------")
# short_data['Anootated tweet'] = short_data['Anootated tweet'].apply(lambda x: re.sub("[^\w\s{P}@;)]+", "", x))
df['Anootated tweet'] = df['Anootated tweet'].apply(lambda x: re.sub("[^\w\s{P}]+", "", x))


raw_data = df['Anootated tweet'].values.tolist()

from sklearn.model_selection import KFold
	
X = df['Anootated tweet'].values
y = df['Class'].values


from sklearn.model_selection import StratifiedKFold
skf = StratifiedKFold(n_splits=10)
skf.get_n_splits(X, y)

tr_vec = []
te_vec = []
y_train = []
y_test = []

for train_index, test_index in skf.split(X, y):

	X_train, X_test = X[train_index], X[test_index]
	ans_train, ans_test = y[train_index], y[test_index]

	y_train.append(ans_train)
	y_test.append(ans_test)

	# c = Counter(ans_test)
	# print(c)



	# if feature_selection=="tfidf_tokenize":
	ngram_range = (1,1)
	tokenizer = TfidfVectorizer(use_idf=True,ngram_range=ngram_range)
	tokenizer.fit(raw_data)

	tr_vec.append(tokenizer.transform(X_train))
	te_vec.append(tokenizer.transform(X_test))
	vocab_size = len(tokenizer.get_feature_names())+1


acc = []
F1 = []
recall = []
precision = []
count = 0

from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import MultinomialNB

for i, j, ans_train, ans_test in zip(tr_vec, te_vec, y_train, y_test):
	# print ('-------- build bernoulli Naive Bayes classifier ------')
	# clf = MultinomialNB()
	# clf.fit(i, ans_train)

	print('------ build svm classifier -------')
	from sklearn.svm import SVC
	clf = SVC(gamma='auto', C=4000)
	# 4000 for Romney
	# 4000 for Obama
	clf.fit(i, ans_train)

	preds = clf.predict(j)


	acc.append(accuracy_score(ans_test, preds))
	# print("accuracy: ", accuracy_score(ans_test, preds))
	# print(clf.score(te_vec, y_test))

	from sklearn.metrics import precision_score
	# print("precision: ", precision_score(ans_test, preds, average=None))
	precision.append(precision_score(ans_test, preds, average=None))

	from sklearn.metrics import recall_score
	# print("recall: ", recall_score(ans_test, preds, average=None))
	recall.append(recall_score(ans_test, preds, average=None))

	from sklearn.metrics import f1_score
	# print("F1 score: ", f1_score(ans_test, preds, average=None))
	F1.append(f1_score(ans_test, preds, average=None))

	print("------- train model --------", count)
	count += 1
	# break


np.asarray(acc)
np.asarray(precision)
np.asarray(recall)
np.asarray(F1)

# print(precision)
print("average accuracy: ", np.average(acc))
print("average precision: ", np.average(precision, axis=0))
print("average recall: ", np.average(recall, axis=0))
print("average F1: ", np.average(F1, axis=0))
