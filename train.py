import tensorflow as tf 
import numpy as np 
import matplotlib.pyplot as plt 
import math
import pickle
import os
import nltk
import json
from model import seq2seq_model,pad_sentence,get_accuracy,sentence_to_seq

from nltk.stem import WordNetLemmatizer
#import config

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
#tf.logging.set_verbosity(tf.logging.ERROR)
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
  
  
lemmatizer = WordNetLemmatizer()
  
# reading the json.intense file
intents = json.loads(open("test_data.json").read())
print(type(intents))
  
# creating empty lists to store data
words = []
classes = []
documents = []
ignore_letters = ["?", "!", ".", ","]
for key, values in intents.items():
    for pattern in intent['patterns']:
        # separating words from patterns
        word_list = nltk.word_tokenize(pattern)
        words.extend(word_list)  # and adding them to words list
          
        # associating patterns with respective tags
        documents.append(((word_list), intent['tag']))
  
        # appending the tags to the class list
        if intent['tag'] not in classes:
            classes.append(intent['tag'])
  
# storing the root words or lemma
words = [lemmatizer.lemmatize(word)
         for word in words if word not in ignore_letters]
words = sorted(set(words))
  
# saving the words and classes list to binary files
pickle.dump(words, open('words.pkl', 'wb'))
pickle.dump(classes, open('classes.pkl', 'wb'))


# we need numerical values of the
# words because a neural network
# needs numerical values to work with
training = []
output_empty = [0]*len(classes)
for document in documents:
	bag = []
	word_patterns = document[0]
	word_patterns = [lemmatizer.lemmatize(
		word.lower()) for word in word_patterns]
	for word in words:
		bag.append(1) if word in word_patterns else bag.append(0)
		
	# making a copy of the output_empty
	output_row = list(output_empty)
	output_row[classes.index(document[1])] = 1
	training.append([bag, output_row])
random.shuffle(training)
training = np.array(training)

# splitting the data
train_x = list(training[:, 0])
train_y = list(training[:, 1])

