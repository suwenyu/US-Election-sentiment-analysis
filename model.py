import pandas as pd
# import matplotlib.pyplot as plt
import re
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

from keras.layers import Dense, SimpleRNN, GRU, LSTM, Embedding, Input, Dropout
from keras.models import Sequential
from keras.models import Model
from keras.utils import to_categorical

from bs4 import BeautifulSoup

sentence_len = 70
import nltk
nltk.data.path.append("/tmp2/wysu/nltk_data")


from keras import backend as K
from keras.engine.topology import Layer
from keras import initializers, regularizers, constraints

from keras.layers import merge, TimeDistributed, Lambda, Flatten, Activation, RepeatVector, Permute, Bidirectional, Reshape

from keras import backend as K
from keras.layers.merge import Dot

class Attention(Layer):
    def __init__(self, step_dim,
                 W_regularizer=None, b_regularizer=None,
                 W_constraint=None, b_constraint=None,
                 bias=True, **kwargs):
        self.supports_masking = True
        self.init = initializers.get('glorot_uniform')

        self.W_regularizer = regularizers.get(W_regularizer)
        self.b_regularizer = regularizers.get(b_regularizer)

        self.W_constraint = constraints.get(W_constraint)
        self.b_constraint = constraints.get(b_constraint)

        self.bias = bias
        self.step_dim = step_dim
        self.features_dim = 0
        super(Attention, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) == 3

        self.W = self.add_weight((input_shape[-1],),
                                 initializer=self.init,
                                 name='{}_W'.format(self.name),
                                 regularizer=self.W_regularizer,
                                 constraint=self.W_constraint)
        self.features_dim = input_shape[-1]

        if self.bias:
            self.b = self.add_weight((input_shape[1],),
                                     initializer='zero',
                                     name='{}_b'.format(self.name),
                                     regularizer=self.b_regularizer,
                                     constraint=self.b_constraint)
        else:
            self.b = None

        self.built = True

    def compute_mask(self, input, input_mask=None):
        return None

    def call(self, x, mask=None):
        features_dim = self.features_dim
        step_dim = self.step_dim

        eij = K.reshape(K.dot(K.reshape(x, (-1, features_dim)),
                        K.reshape(self.W, (features_dim, 1))), (-1, step_dim))

        if self.bias:
            eij += self.b

        eij = K.tanh(eij)

        a = K.exp(eij)

        if mask is not None:
            a *= K.cast(mask, K.floatx())

        a /= K.cast(K.sum(a, axis=1, keepdims=True) + K.epsilon(), K.floatx())

        a = K.expand_dims(a)
        weighted_input = x * a
        return K.sum(weighted_input, axis=1)

    def compute_output_shape(self, input_shape):
        return input_shape[0],  self.features_dim



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



def get_data(feature_selection):

	df = pd.read_excel('trainingObamaRomneytweets.xlsx',sheet_name=0)
	# df = pd.read_excel('trainingObamaRomneytweets.xlsx',sheet_name=1)
	pd.options.display.max_colwidth = 200
	# df.set_index("id", drop=True, append=False, inplace=False, verify_integrity=False)

	#### read the data
	# print (df['Anootated tweet'])
	#
	# print ('--- Print the Basic Info of the data ----')
	# print (df.info())
	# print (df.shape)
	#
	#
	# print ('--- Print the Head/Tail Info of the data ----')
	# print (df.head())
	# print ('--------------------------------------')
	# print (df.tail())


	# df['rate'].plot(kind='hist')
	# plt.show()

	df = df[df['Class'] != '!!!!' ]
	df = df[df['Class'] != 'IR' ]
	df = df[df['Class'] != 'irrelevant']
	df = df[df['Class'] != 'irrevelant']

	df = df[df['Class'] != '' ]
	df = df[df['Class'].notnull()]
	df = df[df['Class'] != 2]
	df = df[df['Class'] != '2']

	short_data = df

	# short_data = df.head(20)
	# print(short_data['Anootated tweet'].to_string(index=False))
	# print(short_data["Anootated tweet"])
	# print(aa)

	short_data['Anootated tweet'] = short_data['Anootated tweet'].apply(lambda x: BeautifulSoup(str(x), 'lxml').get_text())
	# print(short_data['Anootated tweet'].values)
	# print(aa)

	short_data['Anootated tweet'] = short_data['Anootated tweet'].apply(lambda x: re.sub('https?://[A-Za-z0-9./]+', '', x))

	short_data['Anootated tweet'] = short_data['Anootated tweet'].apply(lambda x: re.sub('http?://[A-Za-z0-9./]+', '', x))


	# print(short_data["Anootated tweet"][0])
	




	
	print(short_data.dtypes)
	short_data['Class'] = short_data['Class'].astype(int)
	# short_data['Class'] = short_data['Class'].replace(-1, 3)
	short_data['Class'] = short_data['Class'].replace(-1, 2)

	from collections import Counter
	c = Counter(short_data['Class'].values)
	print(c)
	
	#### remove stop words
	from nltk.corpus import stopwords
	stop = stopwords.words("english")
	# print(short_data['Anootated tweet'].values.tolist())

	print("----------- Remove Stop Word -------------")
	short_data["Anootated tweet"] = short_data["Anootated tweet"].apply(lambda x: ' '.join( word for word in x.split() if word not in stop ))
	# print(short_data['Anootated tweet'].values.tolist())


	#### stemming
	# from nltk.stem import PorterStemmer
	# ps = PorterStemmer()
	# print(short_data['Anootated tweet'].values)
	
	# print('---------- Stemming ---------')
	# short_data['Anootated tweet'] = short_data['Anootated tweet'].apply(lambda x: ' '.join( [ ps.stem(word) for word in x.split() ]))
	# print(short_data['Anootated tweet'].values)



	# #### Lemmatization
	# from nltk.stem.wordnet import WordNetLemmatizer
	# lmtzr = WordNetLemmatizer()
	
	# print("---------- Lemmazation ----------")
	# short_data['Anootated tweet'] = short_data['Anootated tweet'].apply(lambda x: ' '.join([lmtzr.lemmatize(word, 'v') for word in x.split() ]))

	#### lower case
	print ("------ Lower Case -------")
	short_data['Anootated tweet'] = short_data['Anootated tweet'].apply(lambda x: ' '.join(x.lower() for x in x.split()))

	# print(short_data['Anootated tweet'].values)


	#### Clean Twitter
	print ("------ remove punctuation ------")
	# short_data['Anootated tweet'] = short_data['Anootated tweet'].apply(lambda x: re.sub("[^\w\s{P}@;)]+", "", x))
	short_data['Anootated tweet'] = short_data['Anootated tweet'].apply(lambda x: re.sub("[^\w\s{P}]+", "", x))


	# print(short_data['Anootated tweet'].values)

	raw_data = short_data['Anootated tweet'].values.tolist()

	from sklearn.model_selection import KFold
	
	X = short_data['Anootated tweet'].values
	y = short_data['Class'].values

	kf = KFold(n_splits=10)
	
	tr_vec = []
	te_vec = []
	y_train = []
	y_test = []

	for train_index, test_index in kf.split(X):

		X_train, X_test = X[train_index], X[test_index]
		ans_train, ans_test = y[train_index], y[test_index]

		y_train.append(ans_train)
		y_test.append(ans_test)

		# c = Counter(ans_test)
		# print(c)
	# X_train, X_test, y_train, y_test = train_test_split(short_data['Anootated tweet'].values, short_data['Class'].values, test_size=0.2, random_state=0)

		# from collections import Counter
		# c = Counter(short_data['Class'].values)
		# b = Counter(ans_test)
		# print(b)
		# print(c)
	# print(aaa)

		word_index = {}
		print("----------- calculate tokenize ---------")
	
		if feature_selection=="tokenize":
			tokenizer = Tokenizer()
			tokenizer.fit_on_texts(raw_data)
			tr = tokenizer.texts_to_sequences(X_train)
			te = tokenizer.texts_to_sequences(X_test)

			tr_vec.append(pad_sequences(tr, maxlen=sentence_len))
			te_vec.append(pad_sequences(te, maxlen=sentence_len))


			vocab_size = len(tokenizer.word_index)+1
			word_index = tokenizer.word_index
			# print(tr_vec)

		if feature_selection=="tfidf_tokenize":
			ngram_range = (1,1)
			tokenizer = TfidfVectorizer(use_idf=True,ngram_range=ngram_range)
			tokenizer.fit(raw_data)

			tr_vec.append(tokenizer.transform(X_train))
			te_vec.append(tokenizer.transform(X_test))
			vocab_size = len(tokenizer.get_feature_names())+1
		


	return tr_vec, te_vec, y_train, y_test, vocab_size, word_index


def lstmModel(vocab_size, sentence_len, embedding_size, tr_vec, tr_ans, te_vec, te_ans, embed):


	inp = Input(shape=(sentence_len,))
	if len(embed) != 0:
		x = Embedding(vocab_size, embedding_size, weights=[embed])(inp)
	else:
		x = Embedding(input_dim = vocab_size, 
					input_length = sentence_len, 
					output_dim = embedding_size)(inp)

	x = Bidirectional(LSTM(64, return_sequences=False))(x)
	# x = LSTM(128, return_sequences=False)(x)
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

	lstm_em = Dense(32, activation="relu")(x)
	lstm_em = Dropout(0.5)(lstm_em)
	# outp = Dense(3, activation="softmax")(outp)
	outp = Dense(3, activation="softmax")(lstm_em)


	model = Model(inputs=inp, outputs=outp)
	print(model.summary())

	from keras.optimizers import Adam
	adam = Adam(lr=0.001)

	model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])

	model.fit(tr_vec, to_categorical(tr_ans),batch_size=128 ,epochs=2, validation_data=[te_vec, to_categorical(te_ans)], verbose=True)


	print ('--------- make predictions ---------')
	#clf_predictions = clf.predict_proba(te_vec)
	preds = model.predict(te_vec, batch_size=128)
	preds = np.argmax(preds, axis=1)
	# print(accuracy_score(te_ans, preds))
	return preds



if __name__ == '__main__':
	feature_selection = "tokenize"
	# feature_selection = "tfidf_tokenize"
	tr_vec, te_vec, y_train, y_test, vocab_size, word_index = get_data(feature_selection)

	# embedding_size = 300
	

	# predict = lstmModel(vocab_size, sentence_len, embedding_size, tr_vec, y_train, te_vec, y_test, embed)
	# print(tr_vec)

	acc = []
	F1 = []
	recall = []
	precision = []

	count = 0

	embedding_size = 300
	embed = load_vectors('wiki-news-300d-1M.vec', word_index, vocab_size, embedding_size)
	# embed = np.asarray([])

	for i, j, ans_train, ans_test in zip(tr_vec, te_vec, y_train, y_test):


		

		preds = lstmModel(vocab_size, sentence_len, embedding_size, i, ans_train, j, ans_test, embed)
		# print(tr_vec)


		from sklearn.naive_bayes import BernoulliNB
		from sklearn.naive_bayes import MultinomialNB
		print ('-------- build bernoulli Naive Bayes classifier ------')
		# clf = MultinomialNB()
		# clf.fit(i, ans_train)

		# from sklearn.svm import SVC
		# clf = SVC(gamma='auto', C=1500)
		
		# clf.fit(i, ans_train)
		# preds = clf.predict(j)

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
