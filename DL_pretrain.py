import pandas as pd
from tqdm import tqdm
tqdm.pandas()

import numpy as np

import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from bs4 import BeautifulSoup

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

from keras.layers import Dense, SimpleRNN, GRU, LSTM, Embedding, Input, Dropout
from keras.models import Sequential
from keras.models import Model
from keras.utils import to_categorical

from keras import backend as K
from keras.engine.topology import Layer
from keras import initializers, regularizers, constraints

from keras.layers import merge, TimeDistributed, Lambda, Flatten, Activation, RepeatVector, Permute, Bidirectional, Reshape

from keras import backend as K
from keras.layers.merge import Dot

import nltk
nltk.data.path.append("/tmp2/wysu/nltk_data")

sentence_len = 70

# df = pd.read_excel('trainingObamaRomneytweets.xlsx',sheet_name=0)

df = pd.read_csv('train.csv')


# df = df[df['Class'] != '!!!!' ]
# df = df[df['Class'] != 'IR' ]
# df = df[df['Class'] != 'irrelevant']
# df = df[df['Class'] != 'irrevelant']

# df = df[df['Class'] != '' ]
# df = df[df['Class'].notnull()]
# df = df[df['Class'] != 2]
# df = df[df['Class'] != '2']

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


# print('---correct spelling----')
# mispell_dict = {'colour': 'color', 'centre': 'center', 'favourite': 'favorite', 'travelling': 'traveling', 'counselling': 'counseling', 'theatre': 'theater', 'cancelled': 'canceled', 'labour': 'labor', 'organisation': 'organization', 'wwii': 'world war 2', 'citicise': 'criticize', 'youtu ': 'youtube ', 'Qoura': 'Quora', 'sallary': 'salary', 'Whta': 'What', 'narcisist': 'narcissist', 'howdo': 'how do', 'whatare': 'what are', 'howcan': 'how can', 'howmuch': 'how much', 'howmany': 'how many', 'whydo': 'why do', 'doI': 'do I', 'theBest': 'the best', 'howdoes': 'how does', 'mastrubation': 'masturbation', 'mastrubate': 'masturbate', "mastrubating": 'masturbating', 'pennis': 'penis', 'Etherium': 'Ethereum', 'narcissit': 'narcissist', 'bigdata': 'big data', '2k17': '2017', '2k18': '2018', 'qouta': 'quota', 'exboyfriend': 'ex boyfriend', 'airhostess': 'air hostess', "whst": 'what', 'watsapp': 'whatsapp', 'demonitisation': 'demonetization', 'demonitization': 'demonetization', 'demonetisation': 'demonetization'}
# def correct_spelling(x, dic):
#     for word in dic.keys():
#         x = x.replace(word, dic[word])
#     return x

# df['Anootated tweet'] = df['Anootated tweet'].progress_apply(lambda x: correct_spelling(x, mispell_dict))

# from nltk.stem import PorterStemmer
# ps = PorterStemmer()
# # print(short_data['Anootated tweet'].values)

# print('---------- Stemming ---------')
# df['Anootated tweet'] = df['Anootated tweet'].apply(lambda x: ' '.join( [ ps.stem(word) for word in x.split() ]))
# # print(df['Anootated tweet'].values)

df_test = pd.read_csv('test.csv')
df_test['Anootated tweet'] = df_test['Anootated tweet'].apply(str)

df_test['Anootated tweet'] = df_test['Anootated tweet'].apply(lambda x: BeautifulSoup(str(x), 'lxml').get_text())
# print(df['Anootated tweet'].values)
# print(aa)

df_test['Anootated tweet'] = df_test['Anootated tweet'].apply(lambda x: re.sub('https?://[A-Za-z0-9./]+', '', x))

df_test['Anootated tweet'] = df_test['Anootated tweet'].apply(lambda x: re.sub('http?://[A-Za-z0-9./]+', '', x))

#


df_test['Class'] = df_test['Class'].astype(int)
df_test['Class'] = df_test['Class'].replace(-1, 2)


def clean_text(x):

    x = str(x)
    for punct in "/-'":
        x = x.replace(punct, ' ')
    for punct in '&':
        x = x.replace(punct, f' {punct} ')
    for punct in '?!.,"#$%\'()*+-/:;<=>@[\\]^_`{|}~' + '“”’':
        x = x.replace(punct, '')
    return x

df['Anootated tweet'] = df['Anootated tweet'].progress_apply(lambda x: clean_text(x))

df_test['Anootated tweet'] = df_test['Anootated tweet'].progress_apply(lambda x: clean_text(x))






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


	tokenizer = Tokenizer()
	tokenizer.fit_on_texts(raw_data)
	tr = tokenizer.texts_to_sequences(X_train)
	te = tokenizer.texts_to_sequences(X_test)

	tr_vec.append(pad_sequences(tr, maxlen=sentence_len))
	te_vec.append(pad_sequences(te, maxlen=sentence_len))


	vocab_size = len(tokenizer.word_index)+1
	word_index = tokenizer.word_index
	# print(tr_vec)

	# if feature_selection=="tfidf_tokenize":
	# ngram_range = (1,1)
	# tokenizer = TfidfVectorizer(use_idf=True,ngram_range=ngram_range)
	# tokenizer.fit(raw_data)

	# tr_vec.append(tokenizer.transform(X_train))
	# te_vec.append(tokenizer.transform(X_test))
	# vocab_size = len(tokenizer.get_feature_names())+1

X_testing = df_test['Anootated tweet'].values
y_testing = df_test['Class'].values

X_testing_vec = tokenizer.texts_to_sequences(X_testing)
X_testing = pad_sequences(X_testing_vec, maxlen=sentence_len)


def lstmModel(vocab_size, sentence_len, embedding_size, tr_vec, tr_ans, te_vec, te_ans, embed):


	inp = Input(shape=(sentence_len,))
	if len(embed) != 0:
		x = Embedding(vocab_size, embedding_size, weights=[embed])(inp)
	else:
		x = Embedding(input_dim = vocab_size, 
					input_length = sentence_len, 
					output_dim = embedding_size)(inp)

	# x = Bidirectional(LSTM(64, return_sequences=False))(x)
	x = LSTM(128, return_sequences=False)(x)
	# x = Attention(sentence_len)(x)
	


	# attention = Dense(350,input_shape=(70,128,))(x)
	# attention = Activation('tanh')(attention)
	# attention = Dropout(0.5)(attention)
	# attention = Dense(10,input_shape=(70,350))(attention)
	# attention = Activation('softmax')(attention)

	# print(attention.shape)

	# attention = Conv1D(filters=1, kernel_size=350, activation='tanh')(x)
	# attention = Conv1D(filters=350, kernel_size=30, activation='linear')(attention)
	# attention = Lambda(lambda x: K.softmax(x, axis=1), name="attention_vector")(attention)
	# weighted_sequence_embedding = Dot(axes=[1, 1], normalize=False)([attention, x])
	# print(weighted_sequence_embedding.shape)
	# outp = Lambda(lambda x: K.l2_normalize(K.sum(x, axis=1)))(weighted_sequence_embedding)

	lstm_em = Dense(64, activation="relu")(x)
	lstm_em = Dropout(0.5)(lstm_em)
	# outp = Dense(3, activation="softmax")(outp)
	outp = Dense(3, activation="softmax")(lstm_em)


	model = Model(inputs=inp, outputs=outp)
	print(model.summary())

	from keras.optimizers import Adam
	adam = Adam(lr=0.001)

	model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])

	model.fit(tr_vec, to_categorical(tr_ans),batch_size=128 ,epochs=2, validation_split=0.05, verbose=True)


	print ('--------- make predictions ---------')
	#clf_predictions = clf.predict_proba(te_vec)
	preds = model.predict(te_vec, batch_size=128)
	preds = np.argmax(preds, axis=1)
	# print(accuracy_score(te_ans, preds))
	return preds



def load_vectors(fname, word_index, max_words, embed_size):
	emb_mean, emb_std = -0.005838498938828707, 0.4878219664096832
	embedding_matrix = np.random.normal(emb_mean, emb_std, (max_words, embed_size))
	embeddings_index = {}

	with open(fname, 'r', encoding="utf8") as f:
		next(f)
		for line in f:
			values = line.split()
			word = values[0]
			coefs = np.asarray(values[1:], dtype='float32')
			embeddings_index[word] = coefs
	
	embedding_matrix = np.zeros((len(word_index) + 1, embed_size))
	
	for word, i in word_index.items():
		embedding_vector = embeddings_index.get(word)
		if embedding_vector is not None:
		# words not found in embedding index will be all-zeros.
			embedding_matrix[i] = embedding_vector
	return embedding_matrix


acc = []
F1 = []
recall = []
precision = []
count = 0

# from sklearn.naive_bayes import BernoulliNB
# from sklearn.naive_bayes import MultinomialNB

embedding_size = 300
embed = load_vectors('wiki-news-300d-1M.vec', word_index, vocab_size, embedding_size)
# embed = np.asarray([])

for i, j, ans_train, ans_test in zip(tr_vec, te_vec, y_train, y_test):
	
	
	preds = lstmModel(vocab_size, sentence_len, embedding_size, i, ans_train, X_testing, y_testing, embed)
	# print(i)
	# print ('-------- build bernoulli Naive Bayes classifier ------')
	# clf = MultinomialNB()
	# clf.fit(i, ans_train)

	# print('------ build svm classifier -------')
	# from sklearn.svm import SVC
	# clf = SVC(gamma='auto', C=2100)
	# clf.fit(i, ans_train)

	# preds = clf.predict(j)

	# print(preds)	
	acc.append(accuracy_score(y_testing, preds))
	# print("accuracy: ", accuracy_score(ans_test, preds))
	# print(clf.score(te_vec, y_test))

	# from sklearn.metrics import precision_score
	# # print("precision: ", precision_score(ans_test, preds, average=None))
	# precision.append(precision_score(ans_test, preds, average=None))

	# from sklearn.metrics import recall_score
	# # print("recall: ", recall_score(ans_test, preds, average=None))
	# recall.append(recall_score(ans_test, preds, average=None))

	# from sklearn.metrics import f1_score
	# # print("F1 score: ", f1_score(ans_test, preds, average=None))
	# F1.append(f1_score(ans_test, preds, average=None))

	print("------- train model --------", count)
	count += 1
	break


np.asarray(acc)
np.asarray(precision)
np.asarray(recall)
np.asarray(F1)


# print(precision)
f = open('ans_0.txt', 'w')
print("average accuracy: ", np.average(acc))
for index,i in enumerate(preds):
	f.write("%d;;%d\n" % (index, i))
f.close()
# print("average precision: ", np.average(precision, axis=0))
# print("average recall: ", np.average(recall, axis=0))
# print("average F1: ", np.average(F1, axis=0))



