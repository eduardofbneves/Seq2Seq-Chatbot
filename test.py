import config
from utils import preparing_data, make_pred
import pickle
import tensorflow as tf
import json
import numpy as np 
import math
from tqdm import tqdm
import os
from model import seq2seq_model,pad_sentence,get_accuracy,sentence_to_seq
from rouge import Rouge

#import config
#37851 movies

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
#tf.logging.set_verbosity(tf.logging.ERROR)
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

  
# reading the json.intense file as a dict
#json_dict = json.loads(open("test_data.json").read())

path_dir=[]
firstpath= ""
for firstdir in os.listdir('clean_pt'):
    firstpath = "clean_pt/" + firstdir
    for file_dir in os.listdir(firstpath):
        path_dir.append(firstpath + "/" + file_dir)

test_size = round(config.TRAIN_MOVIES*0.35) # percentage
test_movies = math.floor(test_size*config.NMR_MOVIES*0.01)

json_list = []
for movie in range(test_size):
    try:
        with open(path_dir[-movie], encoding='utf8') as jf:
            json_list.append(json.load(jf))
    except FileNotFoundError:
        print("Wrong file path")


BATCH_SIZE = config.BATCH_SIZE
RNN_SIZE = config.RNN_SIZE
EMBED_SIZE = config.EMBED_SIZE
LEARNING_RATE = config.LEARNING_RATE
KEEP_PROB = config.KEEP_PROB
EPOCHS = config.EPOCHS
MODEL_DIR = config.MODEL_DIR
SAVE_PATH = config.SAVE_PATH

max_length = 12
min_length = 1
treshold = 2


questions_int,answers_int,vocabs_to_index,index_to_vocabs,question_vocab_size,answer_vocab_size, questions, answers = preparing_data(json_list,
    max_length, min_length, treshold)

vocab_size = len(index_to_vocabs)

#vocabs_to_index = pickle.load(open("vocab2index.p", "rb"))
#index_to_vocabs = pickle.load(open("index2vocab.p", "rb"))
vocabs_to_index = json.load(open("vocabs/vocab2index.json", encoding='utf8'))
index_to_vocabs = json.load(open("vocabs/index2vocab.json", encoding='utf8'))

batch_size = config.BATCH_SIZE
model_dir = config.MODEL_DIR
save_path = config.SAVE_PATH

loaded_graph = tf.Graph()
sess = tf.InteractiveSession(graph=loaded_graph)
save_path = model_dir+'/'+save_path
loader = tf.train.import_meta_graph(save_path + '.meta')
loader.restore(sess, save_path)
input_data = loaded_graph.get_tensor_by_name('input:0')
logits = loaded_graph.get_tensor_by_name('predictions:0')
input_data_len = loaded_graph.get_tensor_by_name('input_len:0')
target_data_len = loaded_graph.get_tensor_by_name('target_len:0')
keep_prob = loaded_graph.get_tensor_by_name('keep_prob:0')

rouge = Rouge()

for question, answer in zip(questions, answers):
    model_input = sentence_to_seq(question, vocabs_to_index)
    '''
    try:
        output = make_pred(sess,input_data,input_data_len,target_data_len, keep_prob,model_input,batch_size,logits,index_to_vocabs)
    except:
        continue
        '''
    output = make_pred(sess,input_data,input_data_len,target_data_len, keep_prob,model_input,batch_size,logits,index_to_vocabs)
    print(output)
    score = rouge.get_scores(answer, output)