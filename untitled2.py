from matplotlib import pyplot as plt
from model import sentence_to_seq
import json
from sklearn.metrics import plot_confusion_matrix
from tensorflow.math import confusion_matrix

vocabs_to_index = json.load(open("vocabs/vocab2index.json", encoding='utf8'))
sent = "comi peixe hoje"
sent_seq = sentence_to_seq(sent, vocabs_to_index)

pred = "muito peixe bem"
pred_seq = sentence_to_seq(pred, vocabs_to_index)

conf = confusion_matrix(sent_seq, pred_seq)
#plot_confusion_matrix(conf)
print(conf)
