from network import MotionNet
import glob

TEST=True

#The following parameters must have the same value in 'training' and 'test' modes.
num_layer=1
cell='lstm'
hidden_unit=500
time_step = 60
seed_timestep = 30
batch_Frame= 1
frame_time=30
save_period = 20000

'''Execution'''
if TEST :

    MotionNet(TEST=TEST , save_period=save_period , num_layer=num_layer , cell=cell, hidden_unit=hidden_unit , time_step = time_step ,
    seed_timestep = seed_timestep , batch_Frame= batch_Frame , frame_time=frame_time , graphviz=True)


else:

    #batch learning

    completed = MotionNet(epoch=700000 , batch_size=75 , save_period=100000, cost_limit=0.1 ,
    optimizer='adam', learning_rate=0.001 , lr_step=5000, lr_factor=0.99, stop_factor_lr=1e-08 , use_gpu=True ,
    TEST=TEST , num_layer=num_layer , cell=cell , hidden_unit=hidden_unit , time_step = time_step ,
    seed_timestep = seed_timestep , batch_Frame = batch_Frame , frame_time=frame_time , graphviz=True)
    print(completed)

