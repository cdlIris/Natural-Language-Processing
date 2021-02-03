import numpy as np
import re
import pandas as pd
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.metrics import recall_score
from textblob import TextBlob
from tensorflow.keras import layers
from tensorflow import keras
import tensorflow as tf
from sklearn.metrics import precision_score
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier

# This file contains:
# 1. how to label the data
# 2. how to do the tokenize the tweets
# 3. how to one hot encoding the labels
# 4. how to split the train and test sets

# how to label the data
def blob_decide_category(t):
    tb_score = TextBlob(t).sentiment.polarity
    if tb_score < 0:
        return 'negative'
    elif tb_score == 0:
        return 'neutral'
    else:
        return 'positive'
        

# The whole_text.csv contains the tweet content. We can not show it publically by following the restriction of tweepy api.But you can hydrate the content by using the tweet id.

# Change the dir to data/usa/whole_text.csv to proceed usa data
df = pd.read_csv('data/ca/whole_text.csv')
df.columns = ['text']
df['text'] = df['text'].apply(lambda t: str(t))
print(df.head())

df['label'] = df['text'].apply(lambda t: blob_decide_category(t))

# print out negative positive netural tweets
print(df[df['label'] == 'negative'].iloc[0][0])
print(df[df['label'] == 'positive'].iloc[4][0])
print(df[df['label'] == 'neutral'].iloc[3][0])
df['label'].value_counts().plot(kind='bar') # USA
df['label'].value_counts().plot(kind='bar') # CA

# tokenize the tweets
max_features = 2000
tokenizer = Tokenizer(num_words=max_features, split=' ')
tokenizer.fit_on_texts(df['text'].values)
X = tokenizer.texts_to_sequences(df['text'].values)
print(len(X))
X = pad_sequences(X)
print(type(X))

# one hot encoding y
Y = pd.get_dummies(df['label']).values
print(Y.head())

# train and test split
tr_test_split = 0.2
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = tr_test_split, random_state = 42)

# save the matrics
np.save("data/ca/x_tr.npy", X_train)
np.save("data/ca/y_tr.npy", Y_train)
np.save("data/ca/x_test.npy", X_test)
np.save("data/ca/y_test.npy", Y_test)
