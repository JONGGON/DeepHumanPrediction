
from network import MotionNet

TEST=True

#The following parameters must have the same value in 'training' and 'test' modes.
num_layer=1
hidden_unit = 1000
time_step = 90
batch_Frame = 1
save_period = 300
use_gpu=True
use_cudnn=True
Model=2 # # Only 1, 2, and 3 are possible and only works when TEST = True.
'''Execution'''
if TEST:
    MotionNet(TEST=TEST , Model=1 ,save_period=save_period, num_layer=num_layer , hidden_unit=hidden_unit , time_step = time_step , batch_Frame= batch_Frame , use_gpu=use_gpu , use_cudnn=use_cudnn , graphviz=False)
else:
    #batch learning
    MotionNet(epoch=300 , batch_size=68 , save_period=save_period, optimizer='adam', learning_rate=0.01 ,Dropout=0.2 , use_gpu=use_gpu , use_cudnn=use_cudnn ,
    TEST=TEST , num_layer=num_layer , hidden_unit=hidden_unit , time_step = time_step , batch_Frame = batch_Frame ,  graphviz=False )

