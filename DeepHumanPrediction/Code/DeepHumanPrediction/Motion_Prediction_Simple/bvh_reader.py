#-*-coding: utf-8-*-
import numpy as np
import glob
from tqdm import *
import time

def Motion_Data_Preprocessing():

    np.set_printoptions(threshold=1000000)
    files = glob.glob("Data/ACCAD/MartialArts_BVH/*.bvh")
    Data=[]
    time_step=150

    for file_name,i in tqdm(zip(files,range(len(files)))):
        #time.sleep(0.01)
        Raw=[]
        Mdata=[]
        MOTION=False
        
        '''1. basic'''
        print('Processed Data : {}'.format(i+1))
        try:
            with open(file_name, 'r') as f:
                while True:
                    line = f.readline()
                    if line=='MOTION'+'\n' or MOTION: 
                        MOTION=True
                        Raw.append(line)
                    if not line: 
                        break

            for raw in Raw[3:-1]:
                Mdata.append(raw.split())

            '''2. preprocessing for Deeplearning'''
            if len(Mdata) < time_step:
                frame = np.zeros(shape=(time_step-len(Mdata),len(Mdata[0])))
                for i in range(time_step-len(Mdata)):
                    frame[i]=Mdata[-1]
                Mdata = np.concatenate((Mdata,frame), axis=0)
            else :
                Mdata=Mdata[:time_step]
            print(np.shape(Mdata))
            Data.append(Mdata)
        except Exception as e:
            raise e

    '''3. final - data,label '''
    motion_data=list(zip(Data,np.eye(len(Data))))
    xyz_rotation=len(Data[0][0])
    class_number=len(motion_data)
    return motion_data , class_number , time_step , xyz_rotation

if __name__ == "__main__":

    print('Motion_Data_Preprocessing_Starting In Main')
    motion_data , class_number , time_step , xyz_rotation = Motion_Data_Preprocessing()

else:
    print("Motion_Data_Preprocessing_Imported")
