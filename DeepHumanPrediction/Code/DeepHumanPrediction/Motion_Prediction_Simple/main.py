import mxnet as mx
from network import MotionNet

'''implement'''
MotionNet(epoch=1000, batch_size=10, save_period=1000, optimizer='sgd', learning_rate=0.001, use_cudnn=True)