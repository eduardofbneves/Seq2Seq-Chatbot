
from xml.etree import cElementTree as ET
import json

tree = ET.parse('4625179.xml')
root = tree.getroot()

#tree = 
"""document id="3162235">
<s id="1">
<time id="T1S" value="00:01:03,660"/>
<w alternative="Inglaterra" id="1.1">INGLATERRA</w>
<w id="1.2">1554</w>
<time id="T1E" value="00:01:06,458"/>
</s>
<s id="2">
<time id="T2S" value="00:01:09,300"/>
<w alternative="Henrique" id="2.1">HENRIQUE</w>
<w id="2.2">VIII</w>
<w alternative="morreu" id="2.3">MORREU</w>
<time id="T2E" value="00:01:11,734"/>
</s>"""

#root = ET.fromstring(tree)
txt = ""
iter = 1
dict = {}
for word in root.iter('w'):
    
    if word.text.find(',') != -1:
        txt = txt[:-1]
    elif word.text.find('.') != -1: 
        dict.update({'chat {}'.format(iter): txt})
        iter += 1
        txt = ""
        continue
    txt += word.text + " "  

print(dict)

with open("test_data.json", "w") as file:
    my_dict = json.dump(dict, file)