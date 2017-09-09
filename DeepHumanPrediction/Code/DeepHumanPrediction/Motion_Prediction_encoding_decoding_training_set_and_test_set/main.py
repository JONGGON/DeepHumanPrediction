from network import MotionNet
import glob

test=False
time_step=100
seed_timestep=20
batch_Frame=1
epoch=300000
save_period=None
train_file_directory = glob.glob("Data/ACCAD/Transform_Male1_bvh/short_data/train/*.bvh")
'''implement'''

if test==False:
    start_value=1
    geometric_progression=2
    #Sequential learningg
    for i in range(1,len(train_file_directory)+1,1):
        completed,save_period=MotionNet(order = i, epoch=epoch*start_value , batch_size=i, save_period=save_period, optimizer='Adam', learning_rate=0.0008 , lr_step=5000, lr_factor=0.99, stop_factor_lr=1e-08 , use_gpu=True , use_cudnn=True , test=test , predict_size=i  ,time_step = time_step , seed_timestep = seed_timestep , batch_Frame= batch_Frame , frame_time=30)
        print("{}-th data learning".format(i)+completed)
        #geometric series
        start_value*=geometric_progression
else : #test
    MotionNet(order = 2 , epoch=None , batch_size=None , save_period=1360 , optimizer='sgd', learning_rate=0.5 , lr_step=1000, lr_factor=0.99, stop_factor_lr=1e-08 , use_gpu=True , use_cudnn=True , test=test , predict_size=None  , time_step = time_step , seed_timestep = seed_timestep , batch_Frame= batch_Frame , frame_time=30)
