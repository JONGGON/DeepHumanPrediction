import numpy as np
import glob
from tqdm import *
import time
import os
from collections import OrderedDict
import json

def Motion_Data_Preprocessing(time_step = 150 , seed_timestep=30 , batch_Frame=5):

    #cartesian file_preprocessing variables
    np.set_printoptions(threshold=100000, linewidth=100000)
    files=glob.glob("Data_Preprocessing/cartesian_coordinates/*.json")
    joint_name=['Hips', 'LeftUpLeg', 'LeftLeg', 'LeftFoot', 'LeftToeBase', 'LeftToeBaseEnd', 'RightUpLeg', 'RightLeg',
                'RightFoot', 'RightToeBase', 'RightToeBaseEnd', 'Spine', 'Spine1', 'Spine2', 'Spine3', 'Spine4', 'RightShoulder',
                'RightArm', 'RightForeArm', 'RightHand', 'RightHandEnd', 'LeftShoulder', 'LeftArm', 'LeftForeArm', 'LeftHand',
                'LeftHandEnd', 'Neck', 'Head', 'HeadEnd']

    motion_list=[]
    time_step = time_step
    seed_timestep = seed_timestep
    batch_Frame = batch_Frame
    complexity = False
    train_label_motion = []
    file_name = []
    Normalization_factor = 1

    for file, i in tqdm(zip(files, range(len(files)))):
        print('Processed Data : {}'.format(i + 1))
        try:
            with open(file, 'r') as f:
                #1. basic
                data = json.load(f,object_pairs_hook=OrderedDict)
                hips=data['Hips'][0:1]
                spine4=data['Spine4'][0:1]
                for i in range(1,len(list(data.values())[1])):
                    hips=np.concatenate((hips , data['Hips'][3*i:3*i+1]),axis=0)
                    spine4=np.concatenate((spine4 , data['Spine4'][3*i:3*i+1]),axis=0)

                #Prevent duplication
                del data['Hips']
                del data['Spine4']

                data['Hips']=hips
                data['Spine4']=spine4

                #2. concatenate
                motion = np.concatenate(( data['Hips'] , data['LeftUpLeg'] , data['LeftLeg'] , data['LeftFoot'] , data['LeftToeBase'] , data['RightUpLeg'] , data['RightLeg'] ,
                                          data['RightFoot'] , data['RightToeBase'] , data['Spine'] , data['Spine1'] , data['Spine2'] , data['Spine3'], data['Spine4'],
                                          data['RightShoulder'] , data['RightArm'] , data['RightForeArm'] , data['RightHand'] , data['LeftShoulder'], data['LeftArm'],
                                          data['LeftForeArm'] , data['LeftHand'] , data['Neck'] , data['Head']) , axis=1)

                '''motion_data = np.concatenate(( data['Hips'] , data['LeftUpLeg'] , data['LeftLeg'] , data['LeftFoot'] , data['LeftToeBase'] , data['LeftToeBaseEnd'] , data['RightUpLeg'] , data['RightLeg'] ,
                                          data['RightFoot'] , data['RightToeBase'] , data['RightToeBaseEnd'] , data['Spine'] , data['Spine1'] , data['Spine2'] , data['Spine3'], data['Spine4'],
                                          data['RightShoulder'] , data['RightArm'] , data['RightForeArm'] , data['RightHand'] , data['RightHandEnd'] , data['LeftShoulder'], data['LeftArm'],
                                          data['LeftForeArm'] , data['LeftHand'] , data['LeftHandEnd'] , data['Neck'] , data['Head'] , data['HeadEnd']) , axis=1)'''

                motion_list.append(motion)
        except Exception as e:
            raise e

    for i in range(len(motion_list)):
        if complexity:
            motion_list[i]=motion_list[i]*Normalization_factor
        else:
            motion_list[i]=np.floor(motion_list[i])*Normalization_factor

    for i in range(len(files)):
        name=os.path.splitext(os.path.basename(files[i]))[0]
        file_name.append(name)

    #os.path.splitext("path_to_file")[0]
    for i in tqdm(range(len(motion_list))):
        if len(motion_list[i]) < time_step:
            frame = np.zeros(shape=(time_step - len(motion_list[i]), len(motion_list[i][0])))
            for j in range(time_step - len(motion_list[i])):
                frame[j] = motion_list[i][-1]
            motion_list[i] = np.concatenate((motion_list[i], frame), axis=0)
        else:
            motion_list[i] = motion_list[i][:time_step]

    '''3.final -  Motion data preprocessing'''
    # in here, seed_motion : 30_frame , prediction_motion : 120 frame
    for i in range(len(files)):
        train_label_motion.append(motion_list[i][seed_timestep:])

    print("train_seed_motion shape = {}".format(np.shape(motion_list)))
    print("train_label_motion shape = {}".format(np.shape(train_label_motion)))

    train_motion = np.reshape(motion_list,(len(files), int(time_step / batch_Frame), len(motion_list[0][0]) * batch_Frame))
    train_label_motion = np.reshape(train_label_motion, (len(files), int(time_step - seed_timestep) * len(motion_list[0][0])))

    print("-------------------Transform data shape--------------------")
    print("transform_seed_motion shape = {}".format(np.shape(train_motion)))
    print("transform_label_motion shape = {}".format(np.shape(train_label_motion)))

    return Normalization_factor, train_motion, train_label_motion, int(seed_timestep / batch_Frame), int((time_step - seed_timestep) / batch_Frame), len(train_motion[0][0]), file_name

if __name__ == "__main__":

    print('Motion_Data_Preprocessing_Starting In Main')
    Normalization_factor, train_motion, train_label_motion ,seed_timestep , pre_timestep , column , file_directory = Motion_Data_Preprocessing(time_step = 150, seed_timestep=30 , batch_Frame=5)
        
    print("new seed_timestep : {}".format(seed_timestep)) 
    print("new prediction_timestep : {}".format(pre_timestep)) 
    print("new Motion_rotation_data : {}".format(column))

else:
    print("Motion_Data_Preprocessing_Imported")

