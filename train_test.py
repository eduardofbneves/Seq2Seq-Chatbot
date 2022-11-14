import tensorflow as tf 
import numpy as np 
import matplotlib.pyplot as plt 
import math
from tqdm import tqdm
import pickle
import os
import json
from model import seq2seq_model,pad_sentence,get_accuracy,sentence_to_seq

#import config


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
#tf.logging.set_verbosity(tf.logging.ERROR)
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
  
  
# reading the json.intense file as a dict
intents = json.loads(open("test_data.json").read())



