# -*-coding: utf-8-*-
import numpy as np
import glob
from tqdm import *
import sys

def Motion_Data_Preprocessing(time_step=90, batch_Frame=1 ,TEST=False , Model=None):
    
    np.set_printoptions(threshold=1000000)

    if TEST==False:
        file_directory = glob.glob("Dataset/Training/*.bvh")
    else:
        file_directory = glob.glob("Dataset/Test/*.bvh".format(Model))
        if file_directory == list():
            print("Data does not exist")
            sys.exit(0)

    time_step = time_step
    batch_Frame = batch_Frame
    xyz_position = 3
    complexity = False
    Data = []
    motion_label=[]
    motion_number=68
    class_number=12 # Number of human

    '''data normalization'''
    Normalization_factor = 1
    dtype = "int"

    for file_name, i in tqdm(zip(file_directory, range(len(file_directory)))):

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
                        temp = [np.float32(i) * Normalization_factor for i in temp]
                    else:  # complexity = False
                        temp = [np.floor(np.float32(i)) * Normalization_factor for i in temp]
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

    for i in range(len(file_directory)):
        if (i / motion_number) >= 1:
            motion_label.append(int(i/motion_number))
        else:
            motion_label.append(int(i/motion_number))

    print("-------------------Data Shape--------------------")
    print("train_motion shape = {}".format(np.shape(Data)))
    print("\n")

    motion_data = np.reshape(Data, (len(file_directory), int(time_step / batch_Frame), len(Data[0][0]) * batch_Frame))

    print("-------------------Transform Data Shape--------------------")
    print("transform_motion shape = {}".format(np.shape(motion_data)))
    print("\n")

    return Normalization_factor, motion_data , motion_label , int(time_step / batch_Frame) , class_number, len(motion_data[0][0])

if __name__ == "__main__":

    print('Motion_Data_Preprocessing_Starting In Main')
    ormalization_factor, motion_data, motion_label, time_step, class_number, column = Motion_Data_Preprocessing(time_step=90, batch_Frame=1 , TEST=False)

else:
    print("Motion_Data_Preprocessing_Imported")
