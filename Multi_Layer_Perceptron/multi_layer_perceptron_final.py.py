import numpy as np
import math
from gensim.models.keyedvectors import KeyedVectors
from gensim.models import FastText
import nltk
from scipy.sparse import hstack, vstack
from sklearn import preprocessing
from keras.models import Sequential
from keras.layers import (Dense, Embedding, Flatten, BatchNormalization, Dropout, 
                           Input, Activation)
from keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard
from keras.utils import np_utils
from numpy.random import seed
import pickle
import os
import random
import datetime
import warnings
import gc
import urllib.request
import zipfile
import keras
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import (recall_score, precision_score, classification_report, 
                              accuracy_score, confusion_matrix, f1_score)
from sklearn.model_selection import (train_test_split, StratifiedKFold, KFold, 
                                       GridSearchCV, learning_curve, cross_val_score)

training_data = './nlp_train_dataset.txt'

testing_data = './nlp_test_dataset.txt'


def data_processing(fname, include_y=True):
    
    sentences = [] 
    
    with open(fname) as f:
        
        content = f.readlines()
    
    sentence = [] 
    
    for line in content:
        
        if line !='\n':
            
            line = line.strip() 
            
            word = line.split()[0].lower() 
            
            if include_y:
                
                pos = ""
                
                pos = line.split()[1] 
                
                sentence.append((word, pos)) 
            
            else:
                
                sentence.append(word)
        
        else:
            
            sentences.append(sentence) 
            
            sentence = []
    
    return sentences

train_sentences = data_processing(training_data)

test_sentences = data_processing(testing_data, False)


def text_encoding(sentences):
    
    return [[w for w, t in sentence] for sentence in sentences]

def word_to_id(sentences):
    
    wordlist = [item for sublist in text_encoding(sentences) for item in sublist]
    
    word_to_id = {k:v for k,v in enumerate(wordlist)}
    
    return word_to_id

def tagging(tagged_sentence):
    
    return [w for w, _ in tagged_sentence]

def build_vocab(sentences):
    
    vocab =set()
    
    for sentence in sentences:
        
        for word in tagging(sentence):
            
            vocab.add(word)
    
    return sorted(list(vocab))

embs_path = './glove_1M_300d.vec'

embeddings = KeyedVectors.load_word2vec_format(embs_path, binary=False)

w2c = {}

for item in embeddings.key_to_index:
    
    w2c[item] = embeddings.key_to_index[item]


dimensions = embeddings.vectors.shape[1]

padding = np.zeros(dimensions)

np.random.seed(3)

x_padded = np.random.uniform(-0.25, 0.25, dimensions)

def build_embeddings(sentence, index, window=2):
    
    unknown=0
    
    vec = np.array([])
    
    for i in range(index-window, index+window+1):
        
        try:
            
            vec = np.append(vec, embeddings[sentence[i]])
        
        except:
            
            vec = np.append(vec, x_padded)
            
            unknown += 1
    
    return vec, unknown


def set_features(sentence, index):
    
    return {
        'nb_terms': len(sentence),        
        
        'word': sentence[index],
        
        'is_first': index == 0,
        
        'is_last': index == len(sentence) - 1,
        
        'is_capitalized': sentence[index][0].upper() == sentence[index][0],
        
        'is_all_caps': sentence[index].upper() == sentence[index],
        
        'is_all_lower': sentence[index].lower() == sentence[index],
        
        'prefix-1': sentence[index][0],
        
        'prefix-2': sentence[index][:2],
        
        'prefix-3': sentence[index][:3],
        
        'suffix-1': sentence[index][-1],
        
        'suffix-2': sentence[index][-2:],
        
        'suffix-3': sentence[index][-3:],
        
        'i-1_prefix-3': '' if index == 0 else sentence[index-1][:3],        
        
        
        'i-1_suffix-3': '' if index == 0 else sentence[index-1][-3:],        
        
        'i+1_prefix-3': '' if index == len(sentence) - 1 else sentence[index+1][:3],        
        
        'i+1_suffix-3': '' if index == len(sentence) - 1 else sentence[index+1][-3:],        
        
        'prev_word': '' if index == 0 else sentence[index - 1],
        
        'next_word': '' if index == len(sentence) - 1 else sentence[index + 1],
        
        'has_hyphen': '-' in sentence[index],
        
        'is_numeric': sentence[index].isdigit(),
        
        'capitals_inside': sentence[index][1:].lower() != sentence[index][1:],
    
    }

def data_conversion(tagged_sentences, window):
    
    i=0
    
    X, y = [], []
    
    for doc_index, tagged in enumerate(tagged_sentences):
        
        for index in range(len(tagged)):
            
            X.append([set_features(tagging(tagged), index), \
                      build_embeddings(tagging(tagged), index, window)[0], \
                     ])
            
            y.append(tagged[index][1])
            
            k = build_embeddings(tagging(tagged), index, window)[1]
            
            i += k
    
    return X, y, i

def test_conversion(sentence, window):
    
    X = []
    
    for index in range(len(sentence)):
            
            X.append([
                      set_features(sentence, index), \
                      build_embeddings(sentence, index, window), \
                     ])
    return X

def convert_data(train, window=2):
    
    X_train, y_train, unk_tr = data_conversion(train, window=window)
    
    X_train = [x[1] for x in X_train]
    
    X_train = np.asarray(X_train)
    
    return X_train, y_train

X_train, y_train = convert_data(train_sentences)

print(X_train.shape)

classes = sorted(list(set(y_train)))

print(classes)

le = preprocessing.LabelEncoder()

y_train = le.fit_transform(y_train)

y_train = keras.utils.to_categorical(y_train)

print(y_train.shape)

model = Sequential()

model.add(Dense(128, activation='relu', input_dim=X_train.shape[1]))

model.add(Dropout(0.5))

model.add(Dense(64, activation='relu'))

model.add(Dropout(0.5))

model.add(Dense(y_train.shape[1], activation='softmax'))

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

model.summary()

model.fit(X_train, y_train, epochs=10, batch_size=16, verbose=1)


preprocessed_test_data = []

def sentences_embedding(sentence):
    
    X_embs = [x[1][0] for x in sentence]
    
    X_embs = np.asarray(X_embs)
    
    return X_embs

def preprocessing(test_sentences):
    
    for sentence in test_sentences:
        
        sentence = test_conversion(sentence, 2)
        
        embedded = sentences_embedding(sentence)
        
        preprocessed_test_data.append(embedded)

def writing_output(file_name):
    
    with open(file_name, "w") as f:
        
        for sentence in predicted_data:
            
            for word, pos in sentence:
                
                f.write(f"{word} {pos}\n")
            
            f.write(f"\n")

preprocessing(test_sentences)

predicted_data = []

arg_max_dict = []

def predicting(preprocessed_test_data, test_sentences):
    
    for sentence in preprocessed_test_data:
        
        predict_x=model.predict(sentence, batch_size=1, verbose=0) 
        
        predict_x = np.argmax(predict_x, axis=1)
        
        arg_max_dict.append(predict_x)

    for index in range(len(test_sentences)):
        
        predicted_sen = list(zip(test_sentences[index], le.inverse_transform(arg_max_dict[index])))
        
        predicted_data.append(predicted_sen)
predicting(preprocessed_test_data, test_sentences)
writing_output('./output_final.txt')
