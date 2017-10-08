# -*-coding: utf-8-*-
import numpy as np
import glob
from tqdm import *
import os
import sys

def Motion_Data_Preprocessing(time_step=90 , batch_Frame=1):
    
    np.set_printoptions(threshold=1000000)
    file_directory = glob.glob("Model1_Female-Run_turn_right_45.bvh")
    xyz_position = 3
    frame_Time=24
    complexity = True
    Data = []

    '''data normalization'''
    dtype = "int"


    for file_name, i in tqdm(zip(file_directory, range(len(file_directory)))):
        # time.sleep(0.01)
        Raw = []
        Mdata = []
        MOTION = False
        '''1.basic -  Motion data preprocessing'''
        print('Processed Data : {}'.format(i + 1))
        try:
            with open(file_name, 'r') as f:
                while True:
                    line = f.readline()
                    if line == 'MOTION' + "\n" or MOTION:
                        MOTION = True
                        Raw.append(line)
                    if not line:
                        break
            for raw in Raw[3:]:
                # Xposition Yposition Zposition 는 제외
                if dtype == "int":
                    temp = raw.split()[xyz_position:]
                    if complexity:
                        temp = [np.float32(i) for i in temp]
                    else:  # complexity = False
                        temp = [np.floor(np.float32(i)) for i in temp]
                else:  # dtype="str"
                    temp = raw.split()[xyz_position:]
                Mdata.append(temp)
            # Remove the blank line..
            Mdata.pop()

            '''2. Motion data preprocessing - easy for deeplearning'''

            # data padding
            if len(Mdata) < time_step:
                frame = np.zeros(shape=(time_step - len(Mdata), len(Mdata[0])))
                for i in range(time_step - len(Mdata)):
                    frame[i] = Mdata[-1]
                Mdata = np.concatenate((Mdata, frame), axis=0)
            else:
                Mdata = Mdata[:time_step]
            Data.append(Mdata)
        except Exception as e:
            raise e


    print("-------------------Data Shape--------------------")
    print("train_motion shape = {}".format(np.shape(Data)))
    print("\n")

    motion = np.reshape(Data, (len(file_directory), int(time_step / batch_Frame), len(Data[0][0]) * batch_Frame))

    print("-------------------Transform Data Shape--------------------")
    print("transform_motion shape = {}".format(np.shape(motion)))
    print("\n")

    return motion, int(time_step / batch_Frame), batch_Frame , frame_Time ,file_directory

if __name__ == "__main__":

    print('Motion_Data_Preprocessing_Starting In Main')
    motion, time_step, batch_Frame,frame_Time ,file_directory = Motion_Data_Preprocessing(time_step=100, batch_Frame=1)

else:
    print("Motion_Data_Preprocessing_Imported")
