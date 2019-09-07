
# coding: utf-8

# In[1]:


import pandas as pd
from tqdm import tqdm
tqdm.pandas()

df = pd.read_excel('trainingObamaRomneytweets.xlsx',sheet_name=0)

df = df[df['Class'] != '!!!!' ]
df = df[df['Class'] != 'IR' ]
df = df[df['Class'] != 'irrelevant']
df = df[df['Class'] != 'irrevelant']

df = df[df['Class'] != '' ]
df = df[df['Class'].notnull()]
df = df[df['Class'] != 2]
df = df[df['Class'] != '2']

df['Anootated tweet'] = df['Anootated tweet'].apply(str)
print(df.dtypes)
print(df)


# In[2]:


def build_vocab(sentences, verbose =  True):
    """
    :param sentences: list of list of words
    :return: dictionary of words and their count
    """
    vocab = {}
    for sentence in tqdm(sentences, disable = (not verbose)):
        for word in sentence:
            try:
                vocab[word] += 1
            except KeyError:
                vocab[word] = 1
    return vocab

sentences = df['Anootated tweet'].progress_apply(lambda x: x.split()).values
vocab = build_vocab(sentences)
print({k: vocab[k] for k in list(vocab)[:5]})


# In[3]:


from gensim.models import KeyedVectors

news_path = 'wiki-news-300d-1M.vec'
embeddings_index = KeyedVectors.load_word2vec_format(news_path, binary=False)


# In[4]:


import operator 

def check_coverage(vocab,embeddings_index):
    a = {}
    oov = {}
    k = 0
    i = 0
    for word in tqdm(vocab):
        try:
            a[word] = embeddings_index[word]
            k += vocab[word]
        except:

            oov[word] = vocab[word]
            i += vocab[word]
            pass

    print('Found embeddings for {:.2%} of vocab'.format(len(a) / len(vocab)))
    print('Found embeddings for  {:.2%} of all text'.format(k / (k + i)))
    sorted_x = sorted(oov.items(), key=operator.itemgetter(1))[::-1]

    return sorted_x

oov = check_coverage(vocab,embeddings_index)

print(oov[:10])


# In[5]:


'?' in embeddings_index


# In[6]:


'&' in embeddings_index


# In[7]:


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


df['Anootated tweet'] = df['Anootated tweet'].progress_apply(lambda x: clean_text(x))
sentences = df['Anootated tweet'].apply(lambda x: x.split())
vocab = build_vocab(sentences)


# In[9]:


oov = check_coverage(vocab,embeddings_index)
print(oov[:10])


# In[10]:


for i in range(10):
    print(embeddings_index.index2entity[i])


# In[11]:


print(oov[:20])


# In[13]:


import re
def _get_mispell(mispell_dict):
    mispell_re = re.compile('(%s)' % '|'.join(mispell_dict.keys()))
    return mispell_dict, mispell_re


mispell_dict = {'colour':'color',
                'centre':'center',
                'didnt':'did not',
                'doesnt':'does not',
                'isnt':'is not',
                'shouldnt':'should not',
                'favourite':'favorite',
                'travelling':'traveling',
                'counselling':'counseling',
                'theatre':'theater',
                'cancelled':'canceled',
                'labour':'labor',
                'organisation':'organization',
                'wwii':'world war 2',
                'citicise':'criticize',
                'instagram': 'social medium',
                'whatsapp': 'social medium',
                'snapchat': 'social medium'

                }
mispellings, mispellings_re = _get_mispell(mispell_dict)

def replace_typical_misspell(text):
    def replace(match):
        return mispellings[match.group(0)]

    return mispellings_re.sub(replace, text)


# In[16]:


df['Anootated tweet'] = df['Anootated tweet'].progress_apply(lambda x: replace_typical_misspell(x))
sentences = df['Anootated tweet'].progress_apply(lambda x: x.split())
# to_remove = ['a','to','of','and']
sentences = [[word for word in sentence if not word in to_remove] for sentence in tqdm(sentences)]
vocab = build_vocab(sentences)


# In[17]:


oov = check_coverage(vocab,embeddings_index)
print(oov[:20])

