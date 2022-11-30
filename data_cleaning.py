
from os import listdir, path
import shutil

num_dir=[]
path_dir=[]
ind = 0
firstpath= ""
dir = ""
it=0
for firstdir in listdir('pt'):
    firstpath = "pt/" + firstdir
    print(firstdir)
    for lastdir in listdir(firstpath):
        dir = firstpath + "/" + lastdir
        #print(path)
        num_dir = []
        path_dir=[]
        ind = 0
        for file in listdir(dir):
            num_dir.append(path.getsize(dir+ "/" + file))
            path_dir.append(dir+ "/" + file)
        
        ind = num_dir.index(min(num_dir))
        shutil.copyfile(path_dir[ind], "clean_pt/"+firstdir+"/movie{}".format(it))
        it+=1

