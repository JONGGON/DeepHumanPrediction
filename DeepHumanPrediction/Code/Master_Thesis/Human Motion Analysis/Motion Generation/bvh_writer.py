# -*-coding: utf-8-*-
import numpy as np
import os
import shutil

def Motion_Data_Making(seed_motion , pre_motion , seed_timestep , pre_timestep , batch_Frame ,  frame_Time , file_directory, Model=None):
    
    np.set_printoptions(threshold=100000, linewidth=100000)
    test_size=len(seed_motion) # or len(pre_motion)\
    initial_position=np.array([0,93.1,0])
    x_y_z_position=3
    
    seed_timestep=seed_timestep*batch_Frame
    pre_timestep=pre_timestep*batch_Frame
    timestep=seed_timestep+pre_timestep
    motion_length=int(len(seed_motion[0][0])/batch_Frame)

    seed_motion=seed_motion.reshape(test_size , seed_timestep , -1) # -1 = motion_length
    pre_motion=pre_motion.reshape(test_size , pre_timestep , -1) # -1 = motion_length

    all_motion_xyz=np.zeros(shape=(test_size , timestep , motion_length+x_y_z_position)) # basic Xposition Yposition Zposition =  zero

    structure_path="Dataset/Test/Model{}_hierarchy.bvh".format(Model)
    path = "Dataset/Test/Test_Model{}".format(Model)
    if not os.path.exists(path):
        os.makedirs(path)

    #elif TEST==False and not os.path.exists("train_Prediction_motion"):
    #    os.makedirs("train_Prediction_motion")

    #Specify Xposition, Yposition, and Zposition.
    #add Xposition Yposition Zposition 
    for i in range(test_size):

        for j in range(0,seed_timestep,1):
            all_motion_xyz[i][j][:x_y_z_position]=initial_position
            all_motion_xyz[i][j][x_y_z_position:]=seed_motion[i][j]

        for k in range(seed_timestep,timestep,1):
            all_motion_xyz[i][k][:x_y_z_position]=initial_position
            all_motion_xyz[i][k][x_y_z_position:]=pre_motion[i][k-seed_timestep]


    # Write to bvh file - Bvh files of test_size size are generated.
    for i in range(test_size):
        shutil.copy(structure_path,os.path.join(path,os.path.basename(file_directory[i])))
        try:
            with open(os.path.join(path,os.path.basename(file_directory[i])),'a') as file_writer:
                #(1). write frame, frame time
                file_writer.write("\nFrames: {}".format(timestep)+"\n") # fixed
                file_writer.write("Frame Time: {}".format(1/frame_Time)+"\n")

                for j in range(timestep):
                    #(2). Motion data write one line
                    for k in range(motion_length + x_y_z_position):
                        file_writer.write("{0:0.4f} ".format(all_motion_xyz[i][j][k]))
                    file_writer.write("\n")
        except Exception as e:
            raise e


    print("File creation complete")


if __name__ == "__main__":
    print("Do nothing")
else:
    print("Motion_Data_Making_Imported")
