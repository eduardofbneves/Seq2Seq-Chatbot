import tensorflow as tf 
import numpy as np 
import matplotlib.pyplot as plt 
import math
from tqdm import tqdm
import pickle
import os
import json
from model import seq2seq_model,pad_sentence,get_accuracy,sentence_to_seq
import config

#import config


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
#tf.logging.set_verbosity(tf.logging.ERROR)
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

  
# reading the json.intense file as a dict
intents = json.loads(open("test_data.json").read())


BATCH_SIZE = config.BATCH_SIZE
RNN_SIZE = config.RNN_SIZE
EMBED_SIZE = config.EMBED_SIZE
LEARNING_RATE = config.LEARNING_RATE
KEEP_PROB = config.KEEP_PROB
EPOCHS = config.EPOCHS
MODEL_DIR = config.MODEL_DIR
SAVE_PATH = config.SAVE_PATH




