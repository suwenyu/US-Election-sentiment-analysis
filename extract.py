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

sentence_len = 100
import nltk
nltk.data.path.append("/tmp2/wysu/nltk_data")

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

	# short_data['Anootated tweet'] = short_data['Anootated tweet'].apply(lambda x: BeautifulSoup(str(x), 'lxml').get_text())
	# # print(short_data['Anootated tweet'].values)
	# # print(aa)

	# short_data['Anootated tweet'] = short_data['Anootated tweet'].apply(lambda x: re.sub('https?://[A-Za-z0-9./]+', '', x))

	# short_data['Anootated tweet'] = short_data['Anootated tweet'].apply(lambda x: re.sub('http?://[A-Za-z0-9./]+', '', x))


	# print(short_data["Anootated tweet"][0])
	

	# short_data['Class'] = short_data['Class'].replace(-1, 3)
	print(short_data.dtypes)
	short_data['Class'] = short_data['Class'].astype(int)
	short_data['Class'] = short_data['Class'].replace(-1, 2)

	# from collections import Counter
	# c = Counter(short_data['Class'].values)
	# print(c)
	
	# #### remove stop words
	# from nltk.corpus import stopwords
	# stop = stopwords.words("english")
	# # print(short_data['Anootated tweet'].values.tolist())

	# print("----------- Remove Stop Word -------------")
	# short_data["Anootated tweet"] = short_data["Anootated tweet"].apply(lambda x: ' '.join( word for word in x.split() if word not in stop ))
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
	# print ("------ Lower Case -------")
	# short_data['Anootated tweet'] = short_data['Anootated tweet'].apply(lambda x: ' '.join(x.lower() for x in x.split()))

	# # print(short_data['Anootated tweet'].values)


	# #### Clean Twitter
	# print ("------ remove punctuation ------")
	# # short_data['Anootated tweet'] = short_data['Anootated tweet'].apply(lambda x: re.sub("[^\w\s{P}@;)]+", "", x))
	# short_data['Anootated tweet'] = short_data['Anootated tweet'].apply(lambda x: re.sub("[^\w\s{P}]+", "", x))


	# print(short_data['Anootated tweet'].values)

	# raw_data = short_data['Anootated tweet'].values.tolist()
	import numpy as np
	from sklearn.model_selection import train_test_split

	train, test = train_test_split(short_data , test_size=0.33, random_state=42)
	
	test.to_csv('test.csv', index=True)
	train.to_csv('train.csv', index=True)
	# np.savetxt('text.txt', short_data['Anootated tweet'])
	# f = open('text.txt', 'w')
	# for i in raw_data:
	# 	print(i)
		# f.write(i+'\n')
	# f.close()

get_data('tokenize')