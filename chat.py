
#este deve correr bem
from utils import clean, sentence_to_seq, make_pred
import tensorflow as tf
import json
import warnings
import config
warnings.filterwarnings("ignore") # deprecated packages


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

state = True
while state:
	text = input("Escreva a sua mensagem: ") 
	text = clean(text)
	if text == 's':
		state = False
		break
	model_input = sentence_to_seq(text, vocabs_to_index)
	output = make_pred(sess,input_data,input_data_len,target_data_len,
		keep_prob,model_input,batch_size,logits,index_to_vocabs)
	print("Bot: " + output + "\n")