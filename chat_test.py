
#convem treinares antes de correres burro de merda
import config
from utils import *
import pickle
import tensorflow as tf

#vocabs_to_index = pickle.load(open("vocab2index.p", "rb"))
#index_to_vocabs = pickle.load(open("index2vocab.p", "rb"))
vocabs_to_index = json.load(open("vocab2index.json", encoding='utf8'))
index_to_vocabs = json.load(open("index2vocab.json", encoding='utf8'))

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

state = True
while state:
	text = input("Type your message: ") 
	if text == 'q':
		state = False
	model_input = sentence_to_seq(text, vocabs_to_index)
	output = make_pred(sess,input_data,input_data_len,target_data_len,
		keep_prob,model_input,batch_size,logits,index_to_vocabs)
	print("Bot: "+output)
	if text == 'q':
		state = False