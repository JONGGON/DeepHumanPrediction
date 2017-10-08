# -*-coding: utf-8-*-
import numpy as np
import os
import shutil

def Motion_Data_Making(motion , time_step , batch_Frame ,  frame_Time , file_directory):
    
    np.set_printoptions(threshold=100000, linewidth=100000)
    test_size=len(motion) # or len(pre_motion)\
    initial_position=np.array([0,0,0])
    x_y_z_position=3
    
    timestep=time_step*batch_Frame
    motion_length=int(len(motion[0][0])/batch_Frame)

    motion=motion.reshape(test_size , timestep , -1) # -1 = motion_length
    all_motion_xyz=np.zeros(shape=(test_size , timestep , motion_length+x_y_z_position)) # basic Xposition Yposition Zposition =  zero
    structure_path="hierarchy/Model1.bvh"
    #Specify Xposition, Yposition, and Zposition.
    #add Xposition Yposition Zposition 
    for i in range(test_size):

        for j in range(0,timestep,1):
            all_motion_xyz[i][j][:x_y_z_position]=initial_position
            all_motion_xyz[i][j][x_y_z_position:]=motion[i][j]

    # Write to bvh file - Bvh files of test_size size are generated.
    for i in range(test_size):
        shutil.copy(structure_path,os.path.basename(file_directory[i]))
        try:
            with open(os.path.basename(file_directory[i]),'a') as file_writer:
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

    print("File Change complete")

if __name__ == "__main__":
    print("Do nothing")
else:
    print("Motion_Data_Making_Imported")
