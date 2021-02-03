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
file_dir = 'data/usa/'

# change the file_dir to 'data/ca' if want to train Ca data
X_train = np.load(file_dir + "x_tr.npy")
Y_train = np.load(file_dir + "y_tr.npy")
X_test = np.load(file_dir + "x_test.npy")
Y_test = np.load(file_dir + "y_test.npy")


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

# comment the SpatialDropout1D layer to run Naive LSTM
model = keras.Sequential()
model.add(keras.layers.Embedding(max_features, embed_dim, input_length = X_train.shape[1]))
model.add(keras.layers.SpatialDropout1D(dropout_1d))
model.add(keras.layers.LSTM(lstm_out, dropout=dropout_lstm, recurrent_dropout=dropout_rnn))
model.add(keras.layers.Dense(num_class,activation='softmax'))
model.compile(loss = 'categorical_crossentropy', optimizer='adam',metrics = ['accuracy', recall_m, precision_m, f1_m])
print(model.summary())

print("Data shape, x_tr: ", X_train.shape, " x_test: ", X_test.shape)
class AccuracyHistory(keras.callbacks.Callback):
  def on_train_begin(self, logs={}):
      self.tr_csv_fp = 'lstm_tr_log.csv'
      self.test_csv_fp = 'lstm_test_log.csv'

  def on_batch_end(self, batch, logs={}):
      loss, acc, f1, precision, recall = self.model.evaluate(X_test, Y_test, verbose=0)
def tr_log(log):
    with open("tr_log.csv", 'w') as tr_log:
        tr_writer = csv.writer(tr_log)
        tr_writer.writerow([log])

def test_log(log):
    with open("test_log.csv", 'w') as test_log:
        test_writer = csv.writer(test_log)
        test_writer.writerow([log])


import time 


start_time = time.time()

def log_loss(line, fp):
    with open(fp, 'a') as the_file:
        the_file.write(line + '\n')

num_batches = X_train.shape[0] // batch_size
for epoch in np.arange(num_epochs):
    for i in range(num_batches):
        x = X_train[i * batch_size : (i+1) * batch_size]
        y = Y_train[i * batch_size : (i+1) * batch_size]
        loss, acc, precision, recall, f1 = model.train_on_batch(x, y)
        print("[Epoch %d] Batch %d/%d, loss: %f, acc: %05f precision: %05f, recall: %05f, f1: %05f" % (
            epoch+1,i+1, num_batches,loss,acc, precision, recall, f1))

        if i % 50 == 0:
            tr_loss, tr_acc, tr_recall, tr_precision, tr_f1 = model.evaluate(X_train, Y_train, verbose=0)
            test_loss, test_acc, test_recall, test_precision, test_f1 = model.evaluate(X_test, Y_test, verbose=0)
            tr_log = "%d, %d, %f, %05f, %05f, %05f, %05f," % (epoch, i+1, tr_loss, tr_acc, tr_precision, tr_recall, tr_f1)
            test_log = "%d, %d, %f, %05f, %05f, %05f, %05f," % (epoch, i+1, test_loss, test_acc, test_precision, test_recall, test_f1)
            log_loss(tr_log, "tr_log.csv")
            log_loss(test_log, "test_log.csv")
            print("[!] Evalute tr and test at step %d of Epoch %d " % (i+1, epoch))
            print(tr_log)
            print(test_log)
            model_name = "model_" + str(epoch) + "_" + str(i+1)
            model.save(model_name + ".h5")
    tr_loss, tr_acc, tr_recall, tr_precision, tr_f1 = model.evaluate(X_train, Y_train, verbose=0)
    test_loss, test_acc, test_recall, test_precision, test_f1 = model.evaluate(X_test, Y_test, verbose=0)
    tr_log = "%d, %d, %f, %05f, %05f, %05f, %05f," % (epoch, i+1, tr_loss, tr_acc, tr_precision, tr_recall, tr_f1)
    test_log = "%d, %d, %f, %05f, %05f, %05f, %05f," % (epoch, i+1, test_loss, test_acc, test_precision, test_recall, test_f1)
    log_loss(tr_log, "tr_log.csv")
    log_loss(test_log, "test_log.csv")
    print("[!] Evalute tr and test at step %d of Epoch %d " % (i+1, epoch))
    print(tr_log)
    print(test_log)
print("Total time: ", time.time() - start_time)
