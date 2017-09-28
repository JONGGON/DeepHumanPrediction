#Motion Prediction -  2 second prediction completed
from network import MotionNet

TEST=False
Model=3 # Only 1, 2, and 3 are possible and only works when TEST = True.

#The following parameters must have the same value in 'training' and 'test' modes.
num_layer=1
cell='lstm'
hidden_unit = 1000
time_step = 90
seed_timestep = 30 # 0.6 second motion seed  / 2 second motion prediction
batch_Frame = 1
frame_time = 30
save_period = 0
parameter_shared = True # Parameters that determine whether or not the encoder decoder will share parameters

'''Execution'''
if TEST :

    MotionNet(TEST=TEST , save_period=0, num_layer=num_layer , cell=cell, hidden_unit=hidden_unit , time_step = time_step ,
    seed_timestep = seed_timestep , batch_Frame= batch_Frame , frame_time=frame_time , graphviz=True , parameter_shared=parameter_shared , Model=Model)

else:

    #batch learning
    completed = MotionNet(epoch=50000 , batch_size=100 , save_period=save_period, cost_limit=0.01 ,
    optimizer='adam', learning_rate=0.0001 , lr_step=5000, lr_factor=0.99, stop_factor_lr=1e-08 , use_gpu=True ,
    TEST=TEST , num_layer=num_layer , cell=cell , hidden_unit=hidden_unit , time_step = time_step ,
    seed_timestep = seed_timestep , batch_Frame = batch_Frame , frame_time=frame_time , graphviz=True , parameter_shared=parameter_shared)
    print(completed)

