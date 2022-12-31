import os
import json
import numpy as np 
import math
from matplotlib import pyplot as plt
import tensorflow as tf
from tqdm import tqdm
from model import seq2seq_model,pad_sentence,get_accuracy,sentence_to_seq

import config
from utils import preparing_data

#import config
#37851 movies

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
#tf.logging.set_verbosity(tf.logging.ERROR)
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

  
# reading the json.intense file as a dict
path_dir=[]
firstpath= ""
for firstdir in os.listdir('clean_pt'):
    firstpath = "clean_pt/" + firstdir
    for file_dir in os.listdir(firstpath):
        path_dir.append(firstpath + "/" + file_dir)

train_movies = config.TRAIN_MOVIES
print("A carregar {} filmes...".format(train_movies))

json_list = []
for i in range(train_movies):
    movie = round(np.random.random()*config.NMR_MOVIES)
    try:
        with open(path_dir[movie], encoding='utf8') as jf:
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

# boundaries of conversation, 10 words per phrase is already
# quite big for a normal question answer conversation
max_length = 6
min_length = 1
threshold = 2


questions_int,answers_int,vocabs_to_index,index_to_vocabs,question_vocab_size,answer_vocab_size, questions, answers = preparing_data(json_list,
    max_length, min_length, threshold)
print("cerca de {} entradas e {} vocábulos...".format(len(questions_int)*2, len(vocabs_to_index)))

# uncomment to not overwritte everytime
'''
if os.path.exists("vocab2index.json") and os.path.exists("index2vocab.json"):
    print("vocab2index and index2vocab file is present")
else:
    json.dump(vocabs_to_index, open("vocab2index.json", "w", encoding='utf8'), ensure_ascii=False)
    json.dump(index_to_vocabs, open("index2vocab.json", "w", encoding='utf8'), ensure_ascii=False)
'''

json.dump(vocabs_to_index, open("vocabs/vocab2index.json", "w", encoding='utf8'), ensure_ascii=False)
json.dump(index_to_vocabs, open("vocabs/index2vocab.json", "w", encoding='utf8'), ensure_ascii=False)


pad_int = vocabs_to_index['<PAD>']

no_of_batches = math.floor(len(questions_int)//BATCH_SIZE)
round_no = no_of_batches*BATCH_SIZE

input_data,target_data,input_data_len,target_data_len,lr_rate,keep_probs,inference_logits,cost,train_op = seq2seq_model(question_vocab_size,
	EMBED_SIZE,RNN_SIZE,KEEP_PROB,answer_vocab_size,
	BATCH_SIZE,vocabs_to_index)

acc_plt = []
loss_plt = []
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for epoch in range(EPOCHS):
        total_accuracy = 0.0
        total_loss = 0.0
        for bs in tqdm(range(0, round_no, BATCH_SIZE)):
            index = min(bs+BATCH_SIZE, round_no)
            
            batch_x,len_x = pad_sentence(questions_int[bs:index],pad_int)
            batch_y,len_y = pad_sentence(answers_int[bs:index],pad_int)
            batch_x = np.array(batch_x)
            batch_y = np.array(batch_y)
            pred,loss_f,opt = sess.run([inference_logits,cost,train_op], 
                                        feed_dict={input_data:batch_x,
                                                    target_data:batch_y,
                                                    input_data_len:len_x,
                                                    target_data_len:len_y,
                                                    lr_rate:LEARNING_RATE,
                                                    keep_probs:KEEP_PROB})

            train_acc = get_accuracy(batch_y, pred)
            total_loss += loss_f 
            total_accuracy+=train_acc
        
        total_accuracy /= (round_no // BATCH_SIZE)
        total_loss /=  (round_no//BATCH_SIZE)
        acc_plt.append(total_accuracy)
        loss_plt.append(total_loss)
        
        print('Epoch %d, Perda média %f, Exatidão média %f'%(epoch+1, total_loss, total_accuracy))
        print('\n')
        saver = tf.train.Saver() 
        saver.save(sess,MODEL_DIR+"/"+SAVE_PATH)
        
plt.plot(range(len(loss_plt)), loss_plt, color='b', label='perda')
plt.plot(range(len(acc_plt)), acc_plt, color='r', label='exatidão')
plt.title("Perda e Exatidão")
plt.xlabel('Epoch')
plt.legend()
plt.show()    
