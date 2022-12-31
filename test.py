from utils import preparing_data, make_pred, clean
import tensorflow as tf
from tensorflow.math import confusion_matrix
import json
import math
import os
import numpy as np
from rouge import Rouge
from nltk.translate import bleu_score
from model import sentence_to_seq, pad_sentence
import config
import config
warnings.filterwarnings("ignore")

#37851 movies

# avoid tensorflow print on standard error
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

path_dir=[]
firstpath= ""
for firstdir in os.listdir('clean_pt'):
    firstpath = "clean_pt/" + firstdir
    for file_dir in os.listdir(firstpath):
        path_dir.append(firstpath + "/" + file_dir)

test_size = round(config.TRAIN_MOVIES*0.20) # 30 % of train size
print("A carregar {} filmes para treino...".format(test_size))

# getting random movies for training
json_list = []
for i in range(test_size):
    movie = round(np.random.random()*config.NMR_MOVIES)-1
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

# small messages to mimic daily human interaction
max_length = 6
min_length = 1
treshold = 2


questions_int,answers_int,vocabs_to_index,index_to_vocabs,question_vocab_size,answer_vocab_size, questions, answers = preparing_data(json_list,
    max_length, min_length, treshold)

vocab_size = len(index_to_vocabs)

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

score_bleu = []
score_rouge = []
conf_vecs = []
answers_seq = []
pred_seq = []

print("A correr {} perguntas...".format(len(questions)))
for question, answer in zip(questions, answers):
    answer = answer[:-7] # take out the <EOS>
    model_input = sentence_to_seq(clean(question), vocabs_to_index)
    output = make_pred(sess, input_data, input_data_len, target_data_len, 
                       keep_prob, model_input, batch_size,logits, index_to_vocabs)

    score_bleu.append(bleu_score.sentence_bleu(answer, output))
    score_rouge.append(rouge.get_scores(answer, output))

    seqs = pad_sentence([sentence_to_seq(answer, vocabs_to_index), 
                                         sentence_to_seq(output, vocabs_to_index)], vocabs_to_index['<PAD>'])
    answers_seq.append(seqs[0][0])
    pred_seq.append(seqs[0][1])


conf = confusion_matrix(answers_seq, pred_seq, dtype=tf.int32, name=None)
with tf.Session():
    print('Matriz de confusão: \n\n', tf.Tensor.eval(conf,feed_dict=None, session=None)[:10, :10])
    print("Número de palavras corretas (diagonal da matriz): {}".format(np.trace(tf.Tensor.eval(conf,feed_dict=None, session=None))))


print("O valor BLEU médio foi de {} e o valor médio de ROUGE foi {}".format(np.mean(score_bleu), np.mean(score_rouge)))