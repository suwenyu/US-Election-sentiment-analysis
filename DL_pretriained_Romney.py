
# coding: utf-8

# In[1]:


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


# In[2]:


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


# In[3]:


def clean_text(x):

    x = str(x)
    for punct in "/-'":
        x = x.replace(punct, ' ')
    for punct in '&':
        x = x.replace(punct, f' {punct} ')
    for punct in '?!.,"#$%\'()*+-/:;<=>@[\\]^_`{|}~' + '“”’':
        x = x.replace(punct, '')
    return x


# In[8]:


df = pd.read_excel('trainingObamaRomneytweets.xlsx',sheet_name=1)

df['Anootated tweet'] = df['Anootated tweet'].apply(str)
# print(df.dtypes)
# print(df)

df = df[df['Class'] != '!!!!' ]
df = df[df['Class'] != 'IR' ]
df = df[df['Class'] != 'irrelevant']
df = df[df['Class'] != 'irrevelant']

df = df[df['Class'] != '' ]
df = df[df['Class'].notnull()]
df = df[df['Class'] != 2]
df = df[df['Class'] != '2']

df['Class'] = df['Class'].astype(int)
df['Class'] = df['Class'].replace(-1, 2)


from collections import Counter
c = Counter(df['Class'].values)
print(c)


# In[9]:


df['Anootated tweet'] = df['Anootated tweet'].progress_apply(lambda x: clean_text(x))


# In[26]:


# read testing data

df_test = pd.read_csv('Obama_Romney_Test_dataset_NO_label/Romney_Test_dataset_NO_Label.csv', encoding='latin1')
df_test['Tweet_text'] = df_test['Tweet_text'].apply(str)

df_test['Tweet_text'] = df_test['Tweet_text'].progress_apply(lambda x: clean_text(x))


# In[27]:


raw_data = df['Anootated tweet'].values.tolist()+df_test['Tweet_text'].values.tolist()
tokenizer = Tokenizer()
tokenizer.fit_on_texts(raw_data)

vocab_size = len(tokenizer.word_index)+1
word_index = tokenizer.word_index


# In[28]:


# loading word embedding

embedding_size = 300
embed = load_vectors('wiki-news-300d-1M.vec', word_index, vocab_size, embedding_size)


# In[29]:


tr = tokenizer.texts_to_sequences(df['Anootated tweet'].values)
X_training = pad_sequences(tr, maxlen=sentence_len)
y_training = df['Class'].values


X_testing = df_test['Tweet_text'].values
# y_testing = df_test['Class'].values

X_testing_vec = tokenizer.texts_to_sequences(X_testing)
X_testing = pad_sequences(X_testing_vec, maxlen=sentence_len)


# In[37]:


def lstmModel(vocab_size, sentence_len, embedding_size, tr_vec, tr_ans, te_vec, embed):


	inp = Input(shape=(sentence_len,))
	if len(embed) != 0:
		x = Embedding(vocab_size, embedding_size, weights=[embed])(inp)
	else:
		x = Embedding(input_dim = vocab_size, 
					input_length = sentence_len, 
					output_dim = embedding_size)(inp)

	x = LSTM(128, return_sequences=False)(x)
	lstm_em = Dense(64, activation="relu")(x)
	lstm_em = Dropout(0.25)(lstm_em)
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


# In[38]:


preds = lstmModel(vocab_size, sentence_len, embedding_size, X_training, y_training, X_testing, embed)


# In[32]:


f = open('Wen-Yuh_Su_Lanxin_Zhang_Romney.txt', 'w')
# print("average accuracy: ", np.average(acc))
for index,i in enumerate(preds):
    if i == 2:
        i = -1
    f.write("%d;;%d\n" % (index+1, i))
f.close()

