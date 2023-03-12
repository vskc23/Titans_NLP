#Importing libraries
import nltk, re, pprint
import numpy as np
import pandas as pd
import requests
import matplotlib.pyplot as plt
import seaborn as sns
import pprint, time
import random
from sklearn.model_selection import train_test_split
from nltk.tokenize import word_tokenize
import codecs

my_data = []
current_sentence = []
cuurent_word = []

my_file = open('train.txt', 'r')
lines = my_file.readlines()

for line in lines:

    if len(line) <= 1:
        my_data.append(current_sentence)
        current_sentence = []
        continue

    holder = line.split()
    token = holder[0]
    tag = holder[1]

    token_and_tag = []
    token_and_tag.append(token)
    token_and_tag.append(tag)

    current_sentence.append(tuple(token_and_tag))

my_data.append(current_sentence)

#set the random seed
random.seed(1234)

#Splitting into training and test sets
train_set, test_set = train_test_split(my_data,train_size=0.80) # was 95

# Getting list of tagged words in training set
train_tagged_words = [tup for sent in train_set for tup in sent]

# Get length of total tagged words in training set
len(train_tagged_words)

# tokens
tokens = [pair[0] for pair in train_tagged_words]
#tokens[:10]

# vocabulary
V = set(tokens)

# number of pos tags in the training corpus
T = set([pair[1] for pair in train_tagged_words])

# Create numpy array of no of pos tags by total vocabulary
t = len(T)
v = len(V)
w_given_t = np.zeros((t, v))

# Save token and predicted tag to file
def save_to_file(tokens, predicted_tag):
    my_file = open("titans.test.txt","w")

    for x in range(len(predicted_tag)):
        token = tokens[x]
        tag = predicted_tag[x]

        line = str(token) + " " + str(tag) + " \n"

        my_file.write(line)
    
    my_file.close()
    

# compute word given tag: Emission Probability
def word_given_tag(word, tag, train_bag = train_tagged_words):
    tag_list = [pair for pair in train_bag if pair[1]==tag]
    count_tag = len(tag_list)
    w_given_tag_list = [pair[0] for pair in tag_list if pair[0]==word]
    count_w_given_tag = len(w_given_tag_list)
    
    return (count_w_given_tag, count_tag)

# compute tag given tag: tag2(t2) given tag1 (t1), i.e. Transition Probability
def t2_given_t1(t2, t1, train_bag = train_tagged_words):
    tags = [pair[1] for pair in train_bag]
    count_t1 = len([t for t in tags if t==t1])
    count_t2_t1 = 0
    for index in range(len(tags)-1):
        if tags[index]==t1 and tags[index+1] == t2:
            count_t2_t1 += 1
    return (count_t2_t1, count_t1)

# creating t x t (pos tags x pos tags)transition matrix of tags
# each column is t2, each row is t1
# thus M(i, j) represents P(tj given ti)
tags_matrix = np.zeros((len(T), len(T)), dtype='float32')
for i, t1 in enumerate(list(T)):
    for j, t2 in enumerate(list(T)): 
        tags_matrix[i, j] = t2_given_t1(t2, t1)[0]/t2_given_t1(t2, t1)[1]

# convert the matrix to a df for better readability
tags_df = pd.DataFrame(tags_matrix, columns = list(T), index=list(T))
tags_df

#Get total length of tagged words in training corpus
len(train_tagged_words)

# Viterbi Heuristic
def Viterbi(words, train_bag = train_tagged_words):
    #print(words) #tokens
    state = []
    T = list(set([pair[1] for pair in train_bag]))
    
    for key, word in enumerate(words):
        #initialise list of probability column for a given observation
        p = [] 
        for tag in T:
            if key == 0:
                transition_p = tags_df.loc['.', tag]
            else:
                transition_p = tags_df.loc[state[-1], tag]
                
            # compute emission and state probabilities
            emission_p = word_given_tag(words[key], tag)[0]/word_given_tag(words[key], tag)[1]
            state_probability = emission_p * transition_p    
            p.append(state_probability)
            
        pmax = max(p)
        # getting state for which probability is maximum
        state_max = T[p.index(pmax)] 
        state.append(state_max)
    #print(state) #tag
    save_to_file(words, state)
    return list(zip(words, state))

# list of tagged words in test set
test_run_base = [tup for sent in test_set for tup in sent]

# list of  words which are untagged in test set
test_tagged_words = [tup[0] for sent in test_set for tup in sent]

# tagging the test sentences
start = time.time()
tagged_seq = Viterbi(test_tagged_words)
end = time.time()
difference = end-start

#Print total time taken to train the algorithm (seconds)
print("Time: " + str(difference))

# Get accuracy of model
check = [i for i, j in zip(tagged_seq, test_run_base) if i == j] 
accuracy = (len(check)/len(tagged_seq)) * 100
print("Accuracy: " + str(accuracy))