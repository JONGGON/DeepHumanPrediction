import mxnet as mx
import numpy as np
import bvh_reader as br
import bvh_writer as bw
import logging
import os
import time
from collections import OrderedDict
import glob
logging.basicConfig(level=logging.INFO)

print("<<<Motion with encoding decoding>>>")

'''here, use_cudnn=True is faster than use_cudnn=False, but not flexible'''

'''Encoder'''
def encoder(use_cudnn=True,layer_number=2,hidden_number=500,Dropout_rate=0.2):

    cell = mx.rnn.SequentialRNNCell()

    for i in range(layer_number):
        if use_cudnn:
            cell.add(mx.rnn.FusedRNNCell(num_hidden=hidden_number, num_layers=1, bidirectional=False, mode="gru", prefix="gru_encoder_{}".format(i), params=None, forget_bias=1.0, get_next_state=True))
            if Dropout_rate > 0 and (layer_number-1) > i:
                cell.add(mx.rnn.DropoutCell(Dropout_rate, prefix="gru_dropout_encoder_{}".format(i)))
        else:
            cell.add(mx.rnn.LSTMCell(num_hidden=hidden_number, prefix="gru_encoder_{}".format(i)))
            if Dropout_rate > 0 and (layer_number - 1) > i:
                cell.add(mx.rnn.DropoutCell(Dropout_rate, prefix="gru_dropout_encoder_{}".format(i)))

    return cell

'''Decoder'''
def decoder(use_cudnn=True,layer_number=1,hidden_number=500,Dropout_rate=0.2):

    cell = mx.rnn.SequentialRNNCell()

    for i in range(layer_number):

        if use_cudnn:
            cell.add(mx.rnn.FusedRNNCell(num_hidden=hidden_number, num_layers=1, bidirectional=False, mode="gru", prefix="gru__decoder{}".format(i), params=None, forget_bias=1.0, get_next_state=True))
            if Dropout_rate > 0 and (layer_number-1) > i:
                cell.add(mx.rnn.DropoutCell(Dropout_rate, prefix="gru_dropout_decoder_{}".format(i)))

        else:
            cell.add(mx.rnn.LSTMCell(num_hidden=hidden_number, prefix="gru_decoder_{}".format(i)))
            if Dropout_rate > 0 and (layer_number-1) > i:
                cell.add(mx.rnn.DropoutCell(Dropout_rate, prefix="gru_dropout_decoder_{}".format(i)))

    return cell

def MotionNet(order , epoch , batch_size , save_period, optimizer='sgd', learning_rate=0.1 , lr_step=1000, lr_factor=0.9, stop_factor_lr=1e-08 , use_gpu=True , use_cudnn=True , test=False , predict_size=5 ,time_step = 150, seed_timestep=30 , batch_Frame=5 , frame_time=30 ):


    print("-------------------Motion Net-------------------")
    '''1. Data_Loading - bvh_reader'''
    Normalization_factor, train_motion, train_label_motion , seed_timestep, pre_timestep, column, file_directory = br.Motion_Data_Preprocessing(time_step , seed_timestep , batch_Frame)

    if test==True:

        data = OrderedDict()
        data['seed_motion'] = train_motion
        label = {'label_motion': train_label_motion}

        test_iter = mx.io.NDArrayIter(data=data, label=label)
        train_iter = mx.io.NDArrayIter(data=data, label=label , shuffle=False , last_batch_handle='pad')  # Motion data is complex and requires sequential learning to learn from easy examples. So shuffle = False

    else:

        train_motion = train_motion[:predict_size]
        train_label_motion = train_label_motion[:predict_size]

        data = OrderedDict()
        data['seed_motion'] = train_motion
        label = {'label_motion': train_label_motion}

        train_iter = mx.io.NDArrayIter(data=data, label=label, batch_size=batch_size, shuffle=False, last_batch_handle='pad')  # Motion data is complex and requires sequential learning to learn from easy examples. So shuffle = False

    '''2. hyperparameter'''

    rnn_layer_number = 1
    rnn_hidden_number = 500
    fc_number = 500
    Dropout_rate = 0.0

    if use_cudnn == True and use_gpu == True:
        ctx = mx.gpu(0)
    elif use_cudnn == False and use_gpu == True:
        ctx = mx.gpu(0)
    else:
        #why? mx.rnn.FusedRNNCell only works on use_cudnn == True and use_gpu == True
        ctx = mx.cpu(0)

    '''3. Network'''
    previous_output = mx.sym.Variable("previous_output")
    all_motion = mx.sym.Variable('seed_motion')
    seed_motion = mx.sym.transpose(mx.sym.slice_axis(data=all_motion , axis=1 , begin = 0 , end = seed_timestep), axes=(1, 0, 2))  # (time,batch,column)
    label_motion = mx.sym.Variable('label_motion')

    e_cell = encoder(use_cudnn=use_cudnn , layer_number=rnn_layer_number , hidden_number=rnn_hidden_number , Dropout_rate=Dropout_rate)
    d_cell = decoder(use_cudnn=use_cudnn , layer_number=rnn_layer_number , hidden_number=rnn_hidden_number , Dropout_rate=Dropout_rate)
    e_output , e_state = e_cell.unroll(length=seed_timestep , inputs=seed_motion , merge_outputs=False , layout='TNC') #(batch,hidden)

    #seq2seq in test
    if test==True:
        e_output_end = e_output[-1]  # Shape: (1, N, C)
        e_output_end=mx.sym.reshape(data=e_output_end,shape=(1,1,-1))
        d_input = mx.sym.broadcast_to(e_output_end, shape=(pre_timestep, 0, 0))
        d_output , d_state = d_cell.unroll(length=pre_timestep , begin_state=e_state , inputs = d_input, merge_outputs=False, layout='TNC')

    #seq2seq in training 
    else:
        e_output_end = e_output[-1]  # Shape: (1, N, C)
        e_output_end=mx.sym.reshape(data=e_output_end,shape=(1,batch_size,-1))
        d_input = mx.sym.broadcast_to(e_output_end, shape=(pre_timestep, 0, 0))
        d_output , d_state = d_cell.unroll(length=pre_timestep , begin_state=e_state , inputs = d_input, merge_outputs=False, layout='TNC')

    if use_cudnn:
        rnn_output = mx.sym.Reshape(d_output[-1], shape=(-1, rnn_hidden_number))
    else:
        rnn_output = d_output[-1]

    # if you use dropout
    # rnn_output = mx.sym.Dropout(data=rnn_output,p=0.3)
    affine1 = mx.sym.FullyConnected(data=rnn_output , num_hidden=fc_number,name='affine1')  # if use_cudnn=False , data=state[-1] i
    act1 = mx.sym.Activation(data=affine1 , act_type='tanh', name='tanh1')
    drop1 = mx.sym.Dropout(act1 , p=Dropout_rate)
    
    affine2 = mx.sym.FullyConnected(data=drop1 , num_hidden=fc_number , name='affine2') 
    act2 = mx.sym.Activation(data=affine2 , act_type='tanh', name='tanh2')
    drop2 = mx.sym.Dropout(act2 , p=Dropout_rate)

    output = mx.sym.FullyConnected(data=drop2 , num_hidden = pre_timestep*column , name='affine3')
    output = mx.sym.LinearRegressionOutput(data=output , label=label_motion)

    # We visualize the network structure with output size (the batch_size is ignored.) - In pycharm or vs code, not working
    if use_cudnn:
        #shape = {'seed_motion': (seed_timestep, batch_size, column)
        mx.viz.plot_network(symbol=output)  # The diagram can be found on the Jupiter notebook.
    else:
        #shape = {'seed_motion': (batch_size, seed_timestep , column)}
        mx.viz.plot_network(symbol=output)  # The diagram can be found on the Jupiter notebook.

    print(output.list_arguments())

    # training mod
    mod = mx.module.Module(symbol=output, data_names=['seed_motion'], label_names=['label_motion'], context=ctx)

    # Network information print
    print(mod.data_names)
    print(mod.label_names)

    if test==False:
        print(train_iter.provide_data)
        print(train_iter.provide_label)
    else:
        print(test_iter.provide_data)
        print(test_iter.provide_label)

    '''
    grad_req (str, list of str, dict of str to str) – 
    Requirement for gradient accumulation. Can be ‘write’, ‘add’, or ‘null’ 
    (default to ‘write’). Can be specified globally (str) or for each argument (list, dict).
    '''

    mod.bind(data_shapes=train_iter.provide_data , label_shapes=train_iter.provide_label , for_training=True , shared_module=None , inputs_need_grad=False , force_rebind=False , grad_req='write')

    # weights load
    cudnn_weights_path = 'weights/Cudnn_MotionNet-{}th-{}.params'.format(order,save_period)
    weights_path = 'weights/MotionNet-{}th-{}.params'.format(order,save_period)

    if os.path.exists(cudnn_weights_path) and use_cudnn == True and use_gpu==True:
        mod.load_params(cudnn_weights_path)
    elif (os.path.exists(weights_path) and use_cudnn == False and use_gpu==True) and (os.path.exists(weights_path) and use_cudnn == True and use_gpu==False) and  (os.path.exists(weights_path) and use_cudnn == False and use_gpu==False):
        mod.load_params(weights_path)
    else:
        mod.init_params(initializer=mx.initializer.Xavier(rnd_type='gaussian', factor_type='avg', magnitude=1))

    if optimizer=='sgd':
        lr_sch = mx.lr_scheduler.FactorScheduler(step = lr_step, factor = lr_factor , stop_factor_lr = stop_factor_lr)
        mod.init_optimizer(optimizer=optimizer, optimizer_params={'learning_rate': learning_rate , 'lr_scheduler': lr_sch})
    else:
        mod.init_optimizer(optimizer=optimizer, optimizer_params={'learning_rate': learning_rate})

    metric = mx.metric.create(['mse'])

    start_time=time.time()

    if test==False:

        for epoch in range(1, epoch + 1, 1):
            train_iter.reset()
            for batch in train_iter:
                mod.forward(batch, is_train=True)
                mod.update_metric(metric, batch.label)
                mod.backward()
                mod.update()
            print('Epoch : {}'.format(epoch,metric.get()))

            if epoch % 100 == 0:
                end_time=time.time()
                print("-------------------------------------------------------")
                print("{}_learning time : {}".format(epoch,end_time-start_time))
                print("-------------------------------------------------------")

            cal = mod.predict(eval_data=train_iter , merge_batches=True , reset=True, always_output_list=False).asnumpy()
            cost = cal - train_label_motion
            cost=(cost**2)/2
            cost=np.mean(cost)

            print("joint angle Square Error : {}".format(cost))

            if cost < 0.01:

                if not os.path.exists("weights"):
                    os.makedirs("weights")

                if use_cudnn:
                    print('Saving weights')
                    mod.save_params("weights/Cudnn_MotionNet-{}th-{}.params".format(order,epoch))

                else:
                    print('Saving weights')
                    mod.save_params("weights/MotionNet-{}th-{}.params".format(order,epoch))

                print("############################################################################################")
                print("End the learning.")
                print("############################################################################################")

                #training-data_test
                seed = train_iter.data[0][1].asnumpy()
                prediction_motion = mod.predict(eval_data=train_iter, merge_batches=True, reset=True,
                                                        always_output_list=False).asnumpy() / Normalization_factor
                '''Creating a bvh file with predicted values -bvh_writer'''
                bw.Motion_Data_Making(seed[:, :seed_timestep] / Normalization_factor, prediction_motion,
                                              seed_timestep, pre_timestep, batch_Frame, frame_time, file_directory,
                                              test=False)

                return "completed",epoch

        # Network information print
        print(mod.data_shapes)
        print(mod.label_shapes)
        print(mod.output_shapes)
        print(mod.get_params())
        print(mod.get_outputs())
        print("Optimization complete")


    #tall data test
    if test==True:
        # test mod
        test_mod = mx.mod.Module(symbol=output , data_names=['seed_motion'] , label_names=['label_motion'] , context=ctx)
        test_mod.bind(data_shapes=test_iter.provide_data , label_shapes=test_iter.provide_label , for_training=False , shared_module=mod)

        if os.path.exists(cudnn_weights_path) or os.path.exists(weights_path) : #FusedRNN
            seed = test_iter.data[0][1].asnumpy()
            prediction_motion = test_mod.predict(eval_data=test_iter,merge_batches=True ,always_output_list=False).asnumpy()/Normalization_factor

            '''Creating a bvh file with predicted values -bvh_writer'''
            bw.Motion_Data_Making(seed[:,:seed_timestep] , prediction_motion , seed_timestep , pre_timestep , batch_Frame , frame_time , file_directory , test=True)

        else:
            print("Can not test")

    print("finished")

if __name__ == "__main__":
    test = False
    time_step = 100
    seed_timestep = 20
    batch_Frame = 5
    epoch = 5000
    save_period = 0
    file_directory = glob.glob("Data/ACCAD/Transform_Male1_bvh/Short_data/*.bvh")
    print(file_directory)

    if test == False:
        start_value = 1
        geometric_progression = 2
        # Sequential learning
        for i in range(1,len(file_directory)+1,1):
        #for i in range(1, 9, 1):
            print("Data_Number : {}".format(i))
            save_period = MotionNet(order=i,epoch=epoch * start_value, batch_size=i, save_period=save_period, optimizer='adam',
                                    learning_rate=0.1, lr_step=5000, lr_factor=0.99, stop_factor_lr=1e-08, use_gpu=True,
                                    use_cudnn=True, test=test, predict_size=i, time_step=time_step,
                                    seed_timestep=seed_timestep, batch_Frame=batch_Frame, frame_time=30)
            # geometric series
            start_value *= geometric_progression
    else:
        print("MotionNet_imported")