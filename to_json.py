from xml.etree import cElementTree as ET
import json


tree = ET.parse('train_test.xml')
root = tree.getroot()

#root = ET.fromstring(tree)
txt = ""
iter = 1
dict = {}
for word in root.iter('w'):
    
    if word.text.find(',') != -1:
        txt = txt[:-1]
    elif word.text.find('.') != -1: 
        dict.update({'line {}'.format(iter): txt})
        iter += 1
        txt = ""
        continue
    txt += word.text + " "  


with open("test_data.json", "w", encoding='utf8') as file:
    my_dict = json.dump(dict, file, ensure_ascii=False)