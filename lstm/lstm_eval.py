import numpy as np
import re
import csv
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers
from tensorflow import keras 
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, LSTM, SpatialDropout1D

# This file contains how to evaluate the lstm models.
max_features = 2000
tr_test_split = 0.2
num_epochs = 5
embed_dim = 128
lstm_out = 196
dropout_1d = 0.4
dropout_lstm = 0.1
dropout_rnn = 0.2
batch_size = 128
num_class = 3


from tensorflow.keras import backend as K

def recall_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def precision_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))

from tensorflow.python.saved_model import loader_impl
from tensorflow.python.keras.saving.saved_model import load as saved_model_load

##################CA Models####################
# evaluate the ca naive lstm
file_dir = 'data/ca/'
model_dir = "lstm_models/ca_lstm.h5"
X_test = np.load(file_dir + "x_test.npy")
Y_test = np.load(file_dir + "y_test.npy")

model = keras.models.load_model(model_dir, custom_objects={"recall_m": recall_m,
    "precision_m": precision_m,
    "f1_m": f1_m})

print(model.evaluate(X_test, Y_test))

# evaluate the ca lstm with dropout 0.2
file_dir = 'data/ca/'
model_dir = "lstm_models/ca_lstm_2.h5"
X_test = np.load(file_dir + "x_test.npy")
Y_test = np.load(file_dir + "y_test.npy")

model = keras.models.load_model(model_dir, custom_objects={"recall_m": recall_m,
    "precision_m": precision_m,
    "f1_m": f1_m})

print(model.evaluate(X_test, Y_test))

# evaluate the ca lstm with dropout 0.4
file_dir = 'data/ca/'
model_dir = "lstm_models/ca_lstm_4.h5"
X_test = np.load(file_dir + "x_test.npy")
Y_test = np.load(file_dir + "y_test.npy")

model = keras.models.load_model(model_dir, custom_objects={"recall_m": recall_m,
    "precision_m": precision_m,
    "f1_m": f1_m})

print(model.evaluate(X_test, Y_test))


##################USA Models####################
# evaluate the usa naive lstm
file_dir = 'data/usa/'
model_dir = "lstm_models/usa_lstm.h5"
X_test = np.load(file_dir + "x_test.npy")
Y_test = np.load(file_dir + "y_test.npy")

model = keras.models.load_model(model_dir, custom_objects={"recall_m": recall_m,
    "precision_m": precision_m,
    "f1_m": f1_m})

print(model.evaluate(X_test, Y_test))

# evaluate the usa lstm with dropout 0.2
file_dir = 'data/usa/'
model_dir = "lstm_models/usa_lstm_2.h5"
X_test = np.load(file_dir + "x_test.npy")
Y_test = np.load(file_dir + "y_test.npy")

model = keras.models.load_model(model_dir, custom_objects={"recall_m": recall_m,
    "precision_m": precision_m,
    "f1_m": f1_m})

print(model.evaluate(X_test, Y_test))

# evaluate the usa lstm with dropout 0.4
file_dir = 'data/usa/'
model_dir = "lstm_models/usa_lstm_4.h5"
X_test = np.load(file_dir + "x_test.npy")
Y_test = np.load(file_dir + "y_test.npy")

model = keras.models.load_model(model_dir, custom_objects={"recall_m": recall_m,
    "precision_m": precision_m,
    "f1_m": f1_m})

print(model.evaluate(X_test, Y_test))
