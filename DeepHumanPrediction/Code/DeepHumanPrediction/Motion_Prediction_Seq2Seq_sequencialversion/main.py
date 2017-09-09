from network import MotionNet
import glob

train_file_directory = glob.glob("Data/ACCAD/Transform_Male1_bvh/short_data/train/*.bvh") 
TEST=False

#The following parameters must have the same value in 'training' and 'test' modes.
num_layer=1
cell='lstm'
hidden_unit=1000  
time_step = 100 
seed_timestep = 20  
batch_Frame= 1
frame_time=30
save_period=0
'''Execution'''
if TEST :

    MotionNet(TEST=TEST , order=75 , save_period=1 , num_layer=num_layer , cell=cell, hidden_unit=hidden_unit , time_step = time_step ,
    seed_timestep = seed_timestep , batch_Frame= batch_Frame , frame_time=frame_time , graphviz=True)

else:
    start_value=1
    geometric_progression=2

    #Sequential learning

    for i in range(len(train_file_directory),len(train_file_directory)+1,1):

        completed , save_period=MotionNet(order = i , epoch=300000*start_value , batch_size=i , save_period=save_period, cost_limit=1 ,
        optimizer='adam', learning_rate=0.001 , lr_step=5000, lr_factor=0.99, stop_factor_lr=1e-08 , use_gpu=True ,
        TEST=TEST , num_layer=num_layer , cell=cell , hidden_unit=hidden_unit , time_step = time_step ,
        seed_timestep = seed_timestep , batch_Frame = batch_Frame , frame_time=frame_time , graphviz=True)
        
        print("{}-th data learning".format(i)+completed)
        #geometric series
        start_value*=geometric_progression


