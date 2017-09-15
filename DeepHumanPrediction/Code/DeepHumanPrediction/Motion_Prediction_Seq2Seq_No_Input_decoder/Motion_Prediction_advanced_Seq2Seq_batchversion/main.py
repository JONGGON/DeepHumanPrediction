from network import MotionNet

TEST=False

#The following parameters must have the same value in 'training' and 'test' modes.
num_layer=1
cell='lstm'
hidden_unit = 1000
time_step = 80
seed_timestep = 20 # 0.6 second motion seed  / 2 second motion prediction
batch_Frame = 1
frame_time = 30
save_period = 0

'''Execution'''
if TEST :

    MotionNet(TEST=TEST , save_period=1 , num_layer=num_layer , cell=cell, hidden_unit=hidden_unit , time_step = time_step ,
    seed_timestep = seed_timestep , batch_Frame= batch_Frame , frame_time=frame_time , graphviz=True)


else:

    #batch learning

    completed = MotionNet(epoch=700000 , batch_size=75 , save_period=save_period, cost_limit=0.1 ,
    optimizer='adam', learning_rate=0.0003 , lr_step=5000, lr_factor=0.99, stop_factor_lr=1e-08 , use_gpu=True ,
    TEST=TEST , num_layer=num_layer , cell=cell , hidden_unit=hidden_unit , time_step = time_step ,
    seed_timestep = seed_timestep , batch_Frame = batch_Frame , frame_time=frame_time , graphviz=True)
    print(completed)

