import os
import json

i=0
with open('testing and preparation/movie_lines.txt') as topo_file:
    for line in topo_file:
        i+=1
print("Number of lines in original work dataset: {}".format(i))

path_dir=[]
firstpath= ""
for firstdir in os.listdir('clean_pt'):
    firstpath = "clean_pt/" + firstdir
    for file_dir in os.listdir(firstpath):
        path_dir.append(firstpath + "/" + file_dir)
        
lines=343
for movie in range(len(path_dir)):
    try:
        with open(path_dir[movie], encoding='utf8') as jf:
            lines = (lines + len(json.load(jf)))/2
    except FileNotFoundError:
        print("Wrong file path")

print("Average number of lines per .json file: {}".format(lines))
print("Minimum number of movies: {}".format(round(i/lines)+1))
# 455
