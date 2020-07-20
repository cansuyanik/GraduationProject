# -*- coding: utf-8 -*-
"""
Created on Tue Jun  2 19:44:36 2020

@author: yanik
"""

from prepareData import prepareData, getSentence, get_words_and_tags, plot_sentences_long_graph, word_tag_dictionaries
import random
import pandas as pd
import numpy as np
import re
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import os
from keras.utils import get_file
from keras.models import Model, Input
from keras.models import Sequential, load_model
from keras.layers import LSTM, Dense, Dropout,TimeDistributed, Embedding, Masking, Bidirectional
from keras.optimizers import Adam

from keras.utils import plot_model
from keras.callbacks import EarlyStopping, ModelCheckpoint

# dataset file
dataFile = "./data/data.txt"

#vector file
vectorFile = ".//data/cc.tr.300.vec"

#models folder
model_dir = './models/'

# VARIABLES
MAX_LEN = 20  #words limit
LSTM_CELLS = 64
BATCH_SIZE = 32
EPOCH_SIZE = 150
TRAIN_FRACTION = 0.7
LSTM_LAYER = 1
RNN_LAYER = 1
VERBOSE = 0
SAVE_MODEL = True

import sys
def check_sizes(gb_min=1):
    for x in globals():
        size = sys.getsizeof(eval(x)) / 1e9
        if size > gb_min:
            print(f'Object: {x:10}\tSize: {size} GB.')


print(check_sizes(gb_min=1))

os.environ['CUDA_VISIBLE_DEVICES'] = ''
import tensorflow as tf
if tf.test.gpu_device_name():
    print('GPU found')
else:
    print("No GPU found")


# map the sentences to a sequence of numbers and then pad the sequence
# ENDPAD maps to CORR
def map_sentences_and_numbers(sentences, word2idx, tag2idx, n_words, MAX_LEN = 20):
    
    X = [[word2idx[w[0]] for w in s] for s in sentences]
    X = pad_sequences(maxlen=MAX_LEN, sequences=X, padding="post", value=n_words - 1)
    
    y = [[tag2idx[w[1]] for w in s] for s in sentences]
    y = pad_sequences(maxlen=MAX_LEN, sequences=y, padding="post", value=tag2idx["CORR"])
    
    return X, y

#For training, change the labels y to categorial.
def change_labels_to_categorial(y, n_tags):
    
    y = [to_categorical(i, num_classes=n_tags) for i in y]
    return y


#split in train and test set or train, validation and test
def split_dataset(X, y, test_size=0.1):
    
    X_train, X_test, y_train, y_test  = train_test_split(X, y, test_size=test_size)
    return X_train, X_test, y_train, y_test


def createModel(MAX_LEN, n_words, n_tags, embedding_matrix, lstm_cells=LSTM_CELLS,
               trainable=False, lstm_layers=1, bi_direc=False, activation="softmax", 
               optimizer="rmsprop", loss="categorical_crossentropy"):
    
    inputt = Input(shape=(MAX_LEN,))
    if not trainable:
        model = Embedding(input_dim=n_words, output_dim=embedding_matrix.shape[1], 
                          weights=[embedding_matrix],trainable=False, 
                          input_length=MAX_LEN)(inputt)  
    else:
        model = Embedding(input_dim=n_words,
                output_dim=embedding_matrix.shape[1],
                weights=[embedding_matrix],
                trainable=True, input_length=MAX_LEN)(model)
        
    model = Dropout(0.1)(model)
    
    # If want to add multiple LSTM layers
    if lstm_layers > 1:
        for i in range(lstm_layers - 1):
            model = LSTM(units=lstm_cells,
                return_sequences=True,
                recurrent_dropout=0.1)(model)
    
    if bi_direc:
        model = Bidirectional(LSTM(units=lstm_cells, return_sequences=True, 
                               recurrent_dropout=0.1))(model)           # variational biLSTM
    else:
        model = LSTM(units=lstm_cells, return_sequences=True,
                recurrent_dropout=0.1)(model)
        
    out = TimeDistributed(Dense(n_tags, activation=activation))(model)  # softmax output layer
        
    
    model = Model(inputt, out)
    model.compile(optimizer=optimizer, loss=loss, metrics=["accuracy"])
    
    return model
 

def make_callbacks(model_name, save=SAVE_MODEL, monitor="val_loss"):
    """Make list of callbacks for training"""
    callbacks = [EarlyStopping(monitor=monitor, patience=5)]

    if save:
        callbacks.append(
            ModelCheckpoint(
                f'{model_dir}{model_name}.h5',
                save_best_only=True,
                save_weights_only=False))
    return callbacks


def plotGraph(history):
    import matplotlib.pyplot as plt
    hist = pd.DataFrame(history.history)
    plt.figure(figsize=(12,12))
    plt.plot(hist["accuracy"])
    plt.plot(hist["val_accuracy"])
    plt.show()


def load_and_evaluate(X_test, y_test, model_name, return_model=False, batch_size=BATCH_SIZE):
    """Load in a trained model and evaluate with log loss and accuracy"""

    model = load_model(f'{model_dir}{model_name}.h5')
    r = model.evaluate(X_test, np.array(y_test), batch_size=BATCH_SIZE, verbose=1)

    valid_crossentropy = r[0]
    valid_accuracy = r[1]

    print(f'Cross Entropy: {round(valid_crossentropy, 4)}')
    print(f'Accuracy: {round(100 * valid_accuracy, 2)}%')

    if return_model:
        return model


numWords = 0    
def predict(model, words, tags, sentence, word2idx, MAX_LEN, n_words, save=False, outputFile=None):
    
    Words = {}
    sentence = sentence.split()
    global numWords
    numWords = numWords + len(sentence)
    
    sentence2 = []
    
    for i in sentence:
        
        similarity = 0.0
        similarWord = None
        
        if word2idx.get(i) == None:
            print("yes")
            
            for key in word2idx:
                if (key==''):
                    continue
                first = word2vec(key)
                second = word2vec(i)
                
                newSimilarity = cosdis(first, second)
                
                if (newSimilarity > similarity):
                    similarity = newSimilarity
                    similarWord = key    
            print(similarWord)
            sentence2.append(word2idx[similarWord])
            Words[similarWord] = i
        else:
            sentence2.append(word2idx[i])
            Words[i] = i
            
    sentence = [sentence2]   

    #sentence = [[word2idx[i] for i in sentence]]
    sentence = pad_sequences(maxlen=MAX_LEN, sequences=sentence, padding="post", value=n_words - 1)
        
    sentence = sentence[0]
    
    p = model.predict(np.array([sentence]))
    p = np.argmax(p, axis=-1)
    if (save):
        outputFile.write("{:15}: {}".format("Word", "Pred") + "\n")
    print("{:15}: {}".format("Word", "Pred"))
    for w, pred in zip(sentence, p[0]):
        if (words[w] !="ENDPAD"):
            if (save):
                outputFile.write("{:15}: {}".format(Words[words[w]], tags[pred]) + "\n")
            print("{:15}: {}".format(Words[words[w]], tags[pred]))
    
    if(save):
        outputFile.write("\n")
    print("\n")
    
def word2vec(word):
    from collections import Counter
    from math import sqrt

    # count the characters in word
    cw = Counter(word)
    # precomputes a set of the different characters
    sw = set(cw)
    # precomputes the "length" of the word vector
    lw = sqrt(sum(c*c for c in cw.values()))

    # return a tuple
    return cw, sw, lw

def cosdis(v1, v2):
    # which characters are common to the two words?
    common = v1[1].intersection(v2[1])
    # by definition of cosine distance we have
    return sum(v1[0][ch]*v2[0][ch] for ch in common)/v1[2]/v2[2]


DATA = prepareData(dataFile)

words, tags, n_words, n_tags = get_words_and_tags(DATA)
sentences = getSentence(DATA)


word2idx, tag2idx = word_tag_dictionaries(words, tags, MAX_LEN)

X, y = map_sentences_and_numbers(sentences, word2idx, tag2idx, n_words, MAX_LEN)
y = change_labels_to_categorial(y, n_tags)

X_train, X_test, y_train, y_test = split_dataset(X, y)

del sentences


from ELMoForManyLangs.elmoformanylangs import Embedder

# create a weight matrix for words in training docs
# vector dimension 1024
def create_weight_matrix_elmo(word2idx, n_words, vector_dimension = 1024):
    elmoEmbedder = Embedder('./elmo')
    
    notFound = 0
    embedding_matrix = np.zeros((n_words + 1, vector_dimension))
    
    for word, i in word2idx.items():
        
        sent = [[word]]
        embedding_vector = elmoEmbedder.sents2elmo(sent)
        embedding_vector = embedding_vector[0][0]
        #print(word)
        #print(i)
            
        if(embedding_vector is not None):
            embedding_matrix[i] = embedding_vector
        else:
            notFound = notFound + 1
            
    #print('%s words could not found.' % notFound)
    return embedding_matrix

embedding_matrix = create_weight_matrix_elmo(word2idx, n_words)


model_name1 = 'pre-trained-rnn-bi_elmo'
model_name2 = 'pre-trained-rnn-3layer_elmo'
model_name3 = 'pre-trained-rnn-dif_parameters_elmo'

  
    
model = createModel(MAX_LEN, n_words+1, n_tags, embedding_matrix=embedding_matrix, lstm_cells=100, bi_direc=True)
    
model2 = createModel(MAX_LEN, n_words+1, n_tags, embedding_matrix=embedding_matrix, lstm_layers=3, bi_direc=True)
model3 = createModel(MAX_LEN, n_words+1, n_tags, embedding_matrix=embedding_matrix, lstm_cells=256, lstm_layers=3, bi_direc=True)

print(model3.summary()) 

from IPython.display import Image
plot_model(model3, to_file=f'{model_dir}{model_name2}.png', show_shapes=True)

Image(f'{model_dir}{model_name2}.png') 

callbacks = make_callbacks(model_name1)
history = model.fit(X_train, np.array(y_train), batch_size=BATCH_SIZE, epochs=EPOCH_SIZE, validation_split=0.2, verbose=1, callbacks=callbacks)


callbacks2 = make_callbacks(model_name2)
history2 = model2.fit(X_train, np.array(y_train), batch_size=BATCH_SIZE, epochs=EPOCH_SIZE, validation_split=0.2, verbose=1, callbacks=callbacks2)

callbacks3 = make_callbacks(model_name3)
history3 = model3.fit(X_train, np.array(y_train), batch_size=8, epochs=EPOCH_SIZE, validation_split=0.2, verbose=1, callbacks=callbacks3)




plotGraph(history)
model = load_and_evaluate(X_test, y_test, model_name1, return_model=True, batch_size = 32)

plotGraph(history2)
model2 = load_and_evaluate(X_test, y_test, model_name2, return_model=True, batch_size = 32)

plotGraph(history3)
model3 = load_and_evaluate(X_test, y_test, model_name3, return_model=True, batch_size = 8)

exampleSentence = "belkide biz düşünüyor muyuz"
predict(model, words, tags, exampleSentence, word2idx, MAX_LEN, n_words)


file_name1 = "./examples_corr.txt"
file_name2 = "./examples_deda.txt"
file_name3 = "./examples_ki.txt"
file_name4 = "./examples_q.txt"
file_name5 = "./examples_oth.txt"

outputFile = open("./output.txt", "w", encoding="utf8")

file1 = open(file_name1, "r", encoding="utf8")
Lines = file1.readlines() 
for lines in Lines: 
    lines = lines.lower()
    lines = re.sub(r'[^\w\s]','',lines)
    predict(model, words, tags, lines, word2idx, MAX_LEN, n_words, save=True, outputFile= outputFile)

outputFile.close()


