## Obama

#### preprocessing: remove html tag and link, Remove Stop Word, Lower Case, remove punctuation 

1. LSTM
average accuracy:  **0.531593659178919**
average precision:  [0.41656876 0.54731481 0.6108107  0.54031205]
average recall:  [0.40398145 0.52043232 0.67638825 0.52916645]
average F1:  [0.39102067 0.52764319 0.62395138 0.51250317]

==============================

2. NB with tf-idf
average accuracy:  0.46881871235673156
average precision:  [0.4178694  0.55296982 0.56929487 0.47757635]
average recall:  [0.43365101 0.49405616 0.31430457 0.6869806 ]
average F1:  [0.3888523  0.50333977 0.37807332 0.54605833]

==============================
3. SVM (C=100) with tf-idf
average accuracy:  0.203935158130946
average precision:  [0.12669273 0.         0.5        0.07648912]
average recall:  [0.6       0.        0.0042324 0.4      ]
average F1:  [0.20784912 0.         0.00839205 0.12731443]


#### preprocessing: remove html tag and link, Stemming, Lemmazation, Remove Stop Word, Lower Case, remove punctuation 

1. LSTM
average accuracy:  0.5173700552425922
average precision:  [0.41431644 0.53252766 0.63340129 0.54520915]
average recall:  [0.390398   0.56693468 0.65903037 0.53546788]
average F1:  [0.36765195 0.53480579 0.62916965 0.5147527 ]

==============================
2. NB with tf-idf
average accuracy:  0.46254314610068326
average precision:  [0.41620418 0.54392701 0.56123756 0.46968203]
average recall:  [0.42812786 0.47870027 0.30634139 0.69158579]
average F1:  [0.38780033 0.48914016 0.36978804 0.54176565]

==============================
3. SVM (C=100) with tf-idf
average accuracy:  0.203935158130946
average precision:  [0.12669273 0.         0.5        0.07648912]
average recall:  [0.6       0.        0.0042324 0.4      ]
average F1:  [0.20784912 0.         0.00839205 0.12731443]


## Romney

#### preprocessing: remove html tag and link, Remove Stop Word, Lower Case, remove punctuation

1. LSTM
average accuracy:  0.5276566523605151
average precision:  [0.3807161  0.42370991 0.63857934 0.6058972 ]
average recall:  [0.39417395 0.30532747 0.67029327 0.64503984]
average F1:  [0.34809323 0.3442524  0.64022186 0.60489612]

==============================
2. NB with tf-idf
average accuracy:  0.4526396893521357
average precision:  [0.42984479 0.58064115 0.57160348 0.4474445 ]
average recall:  [0.07501319 0.07261725 0.14858561 0.96775575]
average F1:  [0.12020911 0.12522836 0.22386308 0.59860105]

==============================
3. SVM (C=100) with tf-idf
average accuracy:  0.41348763539750666
average precision:  [0.         0.         0.16       0.41344347]
average recall:  [0.         0.         0.00456167 0.99910042]
average F1:  [0.         0.         0.00860504 0.57277914]

#### preprocessing: remove html tag and link, Stemming, Lemmazation, Remove Stop Word, Lower Case, remove punctuation 

1. LSTM
average accuracy:  0.5268040057224607
average precision:  [0.35334798 0.41040041 0.6284358  0.59368915]
average recall:  [0.32963983 0.25069998 0.71455952 0.67546271]
average F1:  [0.31634851 0.26265648 0.65200003 0.61672227]

==============================
2. NB with tf-idf
average accuracy:  0.4514962190884938
average precision:  [0.41604695 0.54969012 0.57201566 0.44850071]
average recall:  [0.08115947 0.06384645 0.14686414 0.9698042 ]
average F1:  [0.12587491 0.11193019 0.22248783 0.59989448]

==============================
3. SVM (C=100) with tf-idf
average accuracy:  0.4132019211117924
average precision:  [0.         0.         0.15714286 0.41328233]
average recall:  [0.         0.         0.00591302 0.99816832]
average F1:  [0.         0.         0.01088664 0.57243884]



====bidrection and pretrained=====
average accuracy:  0.5856014108455592
average precision:  [0.55708345 0.58731479 0.59632273]
average recall:  [0.51480623 0.61821989 0.63081748]
average F1:  [0.51663353 0.59464346 0.59993842]

===attention and pretrained=====
average accuracy:  0.578323735959495
average precision:  [0.53964869 0.58915682 0.58810437]
average recall:  [0.48699185 0.58953973 0.63646522]
average F1:  [0.49515396 0.57769041 0.5959699 ]

=======lstm===========
average accuracy:  0.5543248231702307
average precision:  [0.50807721 0.54943452 0.58344942]
average recall:  [0.50589607 0.60597916 0.54594393]
average F1:  [0.4966207  0.57030388 0.54940809]

=======lstm with pretrain===========
average accuracy:  0.5770864016485149
average precision:  [0.54482173 0.58198316 0.60337451]
average recall:  [0.50528078 0.61106683 0.62533066]
average F1:  [0.502758   0.5891645  0.59495504]

