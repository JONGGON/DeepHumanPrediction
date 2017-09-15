import mxnet as mx
import numpy as np
import bvh_reader as br
import bvh_writer as bw
import customizedRNN as RNN # JG CustomizedRNN.py
import os
import time
from collections import OrderedDict
import logging
logging.basicConfig(level=logging.INFO)

print("<<<Motion with advanced Seq2Seq>>>")
'''Encoder'''
def encoder(layer_number=1,hidden_number=500, Dropout_rate=0.2 , Zoneout_rate=0.0 , cell='gru'):

    print("<<<encoder structure>>>")
    cell_type = cell
    Muilti_cell = mx.rnn.SequentialRNNCell()

    for i in range(layer_number):

        if cell_type == 'gru' or cell_type == 'GRU' or cell_type == 'Gru' :

            if Zoneout_rate > 0:
                print("zoneoutcell applied-{}".format(i))
                Zoneout_cell=mx.rnn.ZoneoutCell(mx.rnn.GRUCell(num_hidden=hidden_number , prefix="gru_encoder_{}".format(i)),zoneout_outputs=Zoneout_rate , zoneout_states=Zoneout_rate)
                Muilti_cell.add(Zoneout_cell)

            else:
                Muilti_cell.add(mx.rnn.GRUCell(num_hidden=hidden_number , prefix="gru_encoder_{}".format(i)))

            print("stack {}-'{}'- decoder cell".format(i,cell_type))

        elif cell_type == 'lstm' or cell_type == 'LSTM' or cell_type == 'Lstm' :


            if Zoneout_rate > 0:
                print("zoneoutcell applied-{}".format(i))
                Zoneout_cell=mx.rnn.ZoneoutCell(mx.rnn.LSTMCell(num_hidden=hidden_number, prefix="lstm_encoder_{}".format(i)),zoneout_outputs=Zoneout_rate , zoneout_states=Zoneout_rate)
                Muilti_cell.add(Zoneout_cell)

            else:
                Muilti_cell.add(mx.rnn.LSTMCell(num_hidden=hidden_number,prefix="lstm_encoder_{}".format(i)))

            print("stack {}-'{}'- decoder cell".format(i,cell_type))

        else:

            if Zoneout_rate > 0:
                print("zoneoutcell applied-{}".format(i))
                Zoneout_cell=mx.rnn.ZoneoutCell(mx.rnn.RNNCell(num_hidden=hidden_number, prefix="rnn_encoder_{}".format(i)),zoneout_outputs=Zoneout_rate , zoneout_states=Zoneout_rate)
                Muilti_cell.add(Zoneout_cell)

            else:
                Muilti_cell.add(mx.rnn.RNNCell(num_hidden=hidden_number, prefix="rnn_encoder_{}".format(i)))

            print("stack {}-'{}'- encoder cell".format(i,cell_type))

        if Dropout_rate > 0 and (layer_number - 1) > i:

            Muilti_cell.add(mx.rnn.DropoutCell(Dropout_rate, prefix="dropout_encoder_{}".format(i)))
            print("stack {}-'{}'- encoder dropout cell".format(i,cell_type))

    print("\n")
    return Muilti_cell

'''Decoder'''
def decoder(layer_number=1 , hidden_number=500 , output_number = 100 , Dropout_rate=0.2 , Zoneout_rate=0.0 , Residual=True ,  cell='gru'):

    print("<<<decoder structure>>>")

    cell_type = cell
    Muilti_cell = RNN.SequentialRNNCell()

    for i in range(layer_number):

        if cell_type == 'gru' or cell_type == 'GRU' or cell_type == 'Gru' :

            if Residual == True and Zoneout_rate > 0:
                print("residualcell applied_{}".format(i))
                Residual_cell=RNN.ResidualCell(RNN.GRUCell(num_hidden=hidden_number, num_output= output_number ,prefix="gru_decoder_{}".format(i)))
                print("zoneoutcell applied-{}".format(i))
                Zoneout_cell=RNN.ZoneoutCell(Residual_cell,zoneout_outputs=Zoneout_rate , zoneout_states=Zoneout_rate)  
                Muilti_cell.add(Zoneout_cell)

            elif Residual == True and Zoneout_rate == 0:
                print("residualcell applied_{}".format(i))
                Residual_cell=RNN.ResidualCell(RNN.GRUCell(num_hidden=hidden_number, num_output= output_number ,prefix="gru_decoder_{}".format(i)))
                Muilti_cell.add(Residual_cell)

            elif Residual == False and Zoneout_rate > 0:
                print("zoneoutcell applied-{}".format(i))
                Zoneout_cell=RNN.ZoneoutCell(RNN.GRUCell(num_hidden=hidden_number, num_output= output_number ,prefix="gru_decoder_{}".format(i)),zoneout_outputs=Zoneout_rate , zoneout_states=Zoneout_rate)  
                Muilti_cell.add(Zoneout_cell)

            else:
                Muilti_cell.add(RNN.GRUCell(num_hidden=hidden_number, num_output= output_number ,prefix="gru_decoder_{}".format(i)))

            print("stack {}-'{}'- decoder cell".format(i,cell_type))

        elif cell_type == 'lstm' or cell_type == 'LSTM' or cell_type == 'Lstm':

            if Residual == True and Zoneout_rate > 0:
                print("residualcell applied_{}".format(i))
                Residual_cell=RNN.ResidualCell(RNN.LSTMCell(num_hidden=hidden_number, num_output=output_number, prefix="lstm_decoder_{}".format(i)))
                print("zoneoutcell applied-{}".format(i))
                Zoneout_cell=RNN.ZoneoutCell(Residual_cell,zoneout_outputs=Zoneout_rate , zoneout_states=Zoneout_rate)  
                Muilti_cell.add(Zoneout_cell)

            elif Residual == True and Zoneout_rate == 0:
                print("residualcell applied_{}".format(i))
                Residual_cell=RNN.ResidualCell(RNN.LSTMCell(num_hidden=hidden_number, num_output=output_number, prefix="lstm_decoder_{}".format(i)))
                Muilti_cell.add(Residual_cell)

            elif Residual == False and Zoneout_rate > 0:
                print("zoneoutcell applied-{}".format(i))
                Zoneout_cell=RNN.ZoneoutCell(RNN.LSTMCell(num_hidden=hidden_number, num_output=output_number, prefix="lstm_decoder_{}".format(i)),zoneout_outputs=Zoneout_rate , zoneout_states=Zoneout_rate)  
                Muilti_cell.add(Zoneout_cell)

            else:
                Muilti_cell.add(RNN.LSTMCell(num_hidden=hidden_number, num_output=output_number, prefix="lstm_decoder_{}".format(i)))

            print("stack {}-'{}'- decoder cell".format(i,cell_type))

        else:

            if Residual == True and Zoneout_rate > 0:
                print("residualcell applied_{}".format(i))
                Residual_cell=RNN.ResidualCell(RNN.RNNCell(num_hidden=hidden_number, num_output=output_number , prefix="rnn_decoder_{}".format(i)))
                print("zoneoutcell applied-{}".format(i))
                Zoneout_cell=RNN.ZoneoutCell(Residual_cell,zoneout_outputs=Zoneout_rate , zoneout_states=Zoneout_rate)  
                Muilti_cell.add(Zoneout_cell)

            elif Residual == True and Zoneout_rate == 0:
                print("residualcell applied_{}".format(i))
                Residual_cell=RNN.ResidualCell(RNN.RNNCell(num_hidden=hidden_number, num_output=output_number,  prefix="rnn_decoder_{}".format(i)))
                Muilti_cell.add(Residual_cell)

            elif Residual == False and Zoneout_rate > 0:
                print("zoneoutcell applied-{}".format(i))
                Zoneout_cell=RNN.ZoneoutCell(RNN.RNNCell(num_hidden=hidden_number, num_output=output_number , prefix="rnn_decoder_{}".format(i)),zoneout_outputs=Zoneout_rate , zoneout_states=Zoneout_rate)
                Muilti_cell.add(Zoneout_cell)

            else:
                Muilti_cell.add(RNN.RNNCell(num_hidden=hidden_number , num_output=output_number, prefix="rnn_decoder_{}".format(i)))

            print("stack {}-'{}'- decoder cell".format(i,cell_type))
            
        if Dropout_rate > 0 and (layer_number-1) > i:
            Muilti_cell.add(RNN.DropoutCell(Dropout_rate, prefix="dropout_decoder_{}".format(i)))
            print("stack {}-'{}'- decoder dropout cell".format(i,cell_type))

    print("\n")
    return Muilti_cell

def MotionNet(epoch=None , batch_size=None , save_period=None , cost_limit=None ,
    optimizer=None, learning_rate=None , lr_step=None , lr_factor=None , stop_factor_lr=None , use_gpu=True ,
    TEST=None , num_layer=None , cell=None, hidden_unit=None ,time_step = None , seed_timestep = None , batch_Frame= None , frame_time=None, graphviz=None):

    print("-------------------Motion Net-------------------")

    '''1. Data_Loading - bvh_reader'''
    Normalization_factor, train_motion, train_label_motion , seed_timestep, pre_timestep, column, file_directory = br.Motion_Data_Preprocessing(time_step , seed_timestep , batch_Frame , TEST)

    if TEST==True:
        print("<TEST>")
        data = OrderedDict()
        data['seed_motion'] = train_motion
        label = {'label_motion': train_label_motion}

        test_iter = mx.io.NDArrayIter(data=data, label=label)

    else:
        print("<Training>")
        data = OrderedDict()
        data['seed_motion'] = train_motion
        label = {'label_motion': train_label_motion}

        train_iter = mx.io.NDArrayIter(data=data, label=label, batch_size=batch_size, shuffle=False, last_batch_handle='pad')  # Motion data is complex and requires sequential learning to learn from easy examples. So shuffle = False ->In here, not using sequential learning

    if use_gpu:
        ctx = mx.gpu(0)
    else:
        ctx = mx.cpu(0)

    '''2. Network'''
    all_motion = mx.sym.Variable('seed_motion')
    label_motion = mx.sym.Variable('label_motion')

    seed_motion = mx.sym.slice_axis(data=all_motion , axis=1 , begin = 0 , end = seed_timestep)  # (batch , time , column)

    if TEST == True:
        pre_motion = mx.sym.reshape(mx.sym.slice_axis(data=all_motion, axis=1, begin=seed_timestep, end=seed_timestep + 1),
                                shape=(1, -1))  # (batch=1,column) - first frame
    else:
        pre_motion = mx.sym.reshape(mx.sym.slice_axis(data=all_motion, axis=1, begin=seed_timestep, end=seed_timestep + 1),
                                shape=(batch_size, -1))  # (batch,column) - first frame


    print("-------------------Network Shape--------------------")
    e_cell = encoder(layer_number= num_layer , hidden_number=hidden_unit , Dropout_rate=0.0 , Zoneout_rate=0.0 , cell=cell)
    d_cell = decoder(layer_number= num_layer , hidden_number=hidden_unit , output_number=column , Dropout_rate=0.0 , Zoneout_rate=0.0 , Residual = True , cell=cell)
    print("\n")

    _ , e_state = e_cell.unroll(length=seed_timestep , inputs=seed_motion , merge_outputs=True , layout='NTC')

    # customized by JG

    if num_layer==1:
        d_output, _ = d_cell.SingleLayer_feed_previous_unroll(length=pre_timestep, begin_state=e_state,inputs=pre_motion, merge_outputs=True, layout='NTC') # MultiLayer_feed_previous_unroll is also possible.
    else:
        d_output, _ = d_cell.MultiLayer_feed_previous_unroll(length=pre_timestep, begin_state=e_state, inputs=pre_motion, merge_outputs=True, layout='NTC') # MultiLayer_feed_previous_unroll is also possible.


    output = mx.sym.LinearRegressionOutput(data = d_output , label=label_motion , grad_scale = 1)

    digraph=mx.viz.plot_network(symbol=output,hide_weights=True)

    #why? batch_Frame>=10 ? -> graph 'plot' size too small for label
    if graphviz==True and TEST == True and batch_Frame>=10:
        digraph.view("{}_batch_Frame_TEST_Seq2Seq".format(batch_Frame)) #show graph
    elif graphviz==True and TEST == False and batch_Frame>=10:
        digraph.view("{}_batch_Frame_Training_Seq2Seq".format(batch_Frame)) #show graph

    print("-------------------Network Learning Parameter--------------------")
    print(output.list_arguments())
    print("\n")

    if TEST == False:
        mod = mx.module.Module(symbol=output, data_names=['seed_motion'], label_names=['label_motion'], context=ctx)
        print("-------------------Network Data Name--------------------")
        print(mod.data_names)
        print(mod.label_names)
        print("\n")
    else:
        # test mod
        test_mod = mx.mod.Module(symbol=output , data_names=['seed_motion'] , label_names=['label_motion'] , context=ctx)
        print("-------------------Network Data Name--------------------")
        print(test_mod.data_names)
        print(test_mod.label_names)
        print("\n")


    print("-------------------Network Data Shape--------------------")
    if TEST==False:
        print(train_iter.provide_data)
        print(train_iter.provide_label)
    else:
        print(test_iter.provide_data)
        print(test_iter.provide_label)
    print("\n")
    '''
    grad_req (str, list of str, dict of str to str) – 
    Requirement for gradient accumulation. Can be ‘write’, ‘add’, or ‘null’ 
    (default to ‘write’). Can be specified globally (str) or for each argument (list, dict).
    '''
    if TEST == False:

        mod.bind(data_shapes=train_iter.provide_data , label_shapes=train_iter.provide_label , for_training=True , shared_module=None , inputs_need_grad=False , force_rebind=False , grad_req='write')
        # weights load
        weights_path = 'weights/MotionNet-{}.params'.format(save_period)

        if os.path.exists(weights_path):
            print("load weights")
            mod.load_params(weights_path)
        else:
            print("init weights")
            mod.init_params(initializer=mx.initializer.Normal(sigma=0.01)) # very important

        start_time=time.time()
        print("-------------------Learning Start--------------------")

        if optimizer=='sgd':
            lr_sch = mx.lr_scheduler.FactorScheduler(step = lr_step, factor = lr_factor , stop_factor_lr = stop_factor_lr)
            mod.init_optimizer(optimizer=optimizer, optimizer_params={'learning_rate': learning_rate , 'lr_scheduler': lr_sch})
        else:
            mod.init_optimizer(optimizer=optimizer, optimizer_params={'learning_rate': learning_rate})

        metric = mx.metric.create(['mse'])

        for epoch in range(1, epoch + 1, 1):
            train_iter.reset()
            metric.reset()
            for batch in train_iter:
                mod.forward(batch, is_train=True)

                #Data Order Transform (N,T,C)
                mod.update_metric(metric,batch.label)

                mod.backward()
                mod.update()

            #print('epoch : {} , MSE : {}'.format(epoch,metric.get()))
            if epoch % 1000 == 0:
                end_time=time.time()
                print("-------------------------------------------------------")
                print("{}_learning time : {}".format(epoch,end_time-start_time))
                print("-------------------------------------------------------")

            if epoch % 10000 == 0:
                if not os.path.exists("weights"):
                    os.makedirs("weights")

                print('Saving weights')
                mod.save_params("weights/MotionNet-{}.params".format(epoch))

            cal = mod.predict(eval_data=train_iter ,  merge_batches=True , reset=True, always_output_list=False).asnumpy() / Normalization_factor
            cost = cal - train_label_motion
            cost=(cost**2)/2
            cost=np.mean(cost)
            print('{} epoch '.format(epoch), end='')
            print("Joint Angle Square Error : {}".format(cost))

            if cost < cost_limit:

                if not os.path.exists("weights"):
                    os.makedirs("weights")

                print('Saving weights')
                mod.save_params("weights/MotionNet-{}.params".format(epoch))

                print("############################################################################################")
                print("End the learning.")
                print("############################################################################################")

                #order : (N , T(all_time) , C)
                seed = train_iter.data[0][1].asnumpy()
                #order : (N , T(predict time) , C)
                prediction_motion = mod.predict(eval_data=train_iter, merge_batches=True, reset=True, always_output_list=False).asnumpy() / Normalization_factor

                '''Creating a bvh file with predicted values -bvh_writer'''
                bw.Motion_Data_Making(seed[:,:seed_timestep] / Normalization_factor, prediction_motion,
                                              seed_timestep, pre_timestep, batch_Frame, frame_time, file_directory,
                                              TEST)

                return "optimization completed"

        print("\n")

        print("-------------------Network Information--------------------")
        print(mod.data_shapes)
        print(mod.label_shapes)
        print(mod.output_shapes)
        print(mod.get_params())
        print(mod.get_outputs())
        print("{} learning optimization completed".format(epoch))
        print("\n")

    if TEST==True:

        test_mod.bind(data_shapes=test_iter.provide_data , label_shapes=test_iter.provide_label , for_training=False)

        # weights load
        weights_path = 'weights/MotionNet-{}.params'.format(save_period)
        if os.path.exists(weights_path):
            test_mod.load_params(weights_path)

            #order : (N , T(all time) , C)
            seed = test_iter.data[0][1].asnumpy()

            #order : (N , T(predict time) , C)
            prediction_motion = test_mod.predict(eval_data=test_iter , merge_batches=True , always_output_list=False).asnumpy()/Normalization_factor

            print("Test Prediction motion shape : {}".format(np.shape(prediction_motion)))

            #test cost
            cost = prediction_motion - train_label_motion
            cost=(cost**2)/2
            cost=np.mean(cost)
            print("prediction error : {}".format(cost))

            '''Creating a bvh file with predicted values -bvh_writer'''
            bw.Motion_Data_Making(seed[:,:seed_timestep] / Normalization_factor , prediction_motion , seed_timestep , pre_timestep , batch_Frame , frame_time , file_directory ,TEST)

            return "learning completed"

        else:

            print("Can not test")


if __name__ == "__main__":

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

        completed=MotionNet(TEST=TEST , save_period=1 , num_layer=num_layer , cell=cell, hidden_unit=hidden_unit , time_step = time_step , seed_timestep = seed_timestep , batch_Frame= batch_Frame , frame_time=frame_time ,graphviz=True)
        print(completed)

    else:

        completed = MotionNet(epoch=300000 , batch_size=5 , save_period=save_period, cost_limit=0.1 ,
        optimizer='adam', learning_rate=0.001 , lr_step=5000, lr_factor=0.99, stop_factor_lr=1e-08 , use_gpu=True ,
        TEST=TEST , num_layer=num_layer , cell=cell , hidden_unit=hidden_unit , time_step = time_step , seed_timestep = seed_timestep , batch_Frame = batch_Frame , frame_time=frame_time , graphviz=True)
        print(completed)

else:
    print("MotionNet_imported")