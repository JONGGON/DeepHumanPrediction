import os
import shutil
import glob
import sys

#T = "Training" #Or "Test"
T = "Training"
model = "Model19"
data_path="orginal/{}/{}/bvh/*bvh".format(T,model)
rename_path = "rename/{}/{}".format(T,model)

files=glob.glob("orginal/{}/{}/bvh/*bvh".format(T,model))

if files == list():
    print("Data does not exist")
    sys.exit(0)

if not os.path.exists(rename_path):
    os.makedirs(rename_path)

for i in range(len(files)):
    shutil.copy(files[i],os.path.join(rename_path,model+"_"+os.path.basename(files[i]))) # copy!!!
    #os.rename(files[i],os.path.join(path,model+"_"+os.path.basename(files[i]))) #  move!!
print("from {} to {} Copy is Completed".format(os.path.dirname(files[0]), rename_path))
#print("from {} to {} Copy is Completed".format(os.path.dirname(os.path.abspath(files[0])),os.path.abspath(rename_path)))

