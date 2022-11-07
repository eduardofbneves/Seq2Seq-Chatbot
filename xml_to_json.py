
from xml.etree import cElementTree as ET
import json
from os import walk, listdir
from os.path import isfile, join

'''
firstpath= ""
path = ""
for firstdir in listdir('pt'):
    firstpath = "pt/" + firstdir
    for lastdir in listdir(firstpath):
        path = firstpath + "/" + lastdir
        for file in listdir(path):
            dir.append(path + "/"+ file)
'''

dir = []
for root, dirs, files in walk("pt"):
    for file in files:
        dir.append(root + "/" + file)


for xml in range(0, len(dir)):
    tree = ET.parse(dir[xml])
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


    with open("Dataset/data{}.json".format(xml), "w", encoding='utf8') as file:
        my_dict = json.dump(dict, file, ensure_ascii=False)