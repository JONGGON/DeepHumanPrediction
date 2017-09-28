import os
import shutil
import glob
import sys

rename_path = "rename_Training" # "rename_Test"
#rename_path = "rename_Test"
files=glob.glob("Training/*bvh")
#files=glob.glob("Test/*bvh")

if not os.path.exists(rename_path):
    os.makedirs(rename_path)

with open(os.path.join(rename_path,"Motion list.txt"),'w') as f:
    f.write("<Data List>\n")
    for name in files:
        f.write(os.path.basename(name)+"\n")

if files == list():
    print("Data does not exist")
    sys.exit(0)

for i in range(len(files)):
    shutil.copy(files[i],os.path.join(rename_path,"{}.bvh".format(i))) # copy!!!
    #os.rename(files[i],os.path.join(path,model+"_"+os.path.basename(files[i]))) #  move!!
print("from {} to {} Copy is Completed".format(os.path.dirname(files[0]), rename_path))
#print("from {} to {} Copy is Completed".format(os.path.dirname(os.path.abspath(files[0])),os.path.abspath(rename_path)))

