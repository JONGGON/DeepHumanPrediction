import mxnet as mx
import numpy as np
import matplotlib.pyplot as plt
import bvh_reader as br
import logging
import os
logging.basicConfig(level=logging.INFO)

def MotionNet(epoch,batch_size,save_period, optimizer='sgd', learning_rate=0.001 , use_cudnn = True):

    rnn_hidden_number = 500
    fc_number=200
    optimizer=optimizer
    learning_rate=learning_rate
    use_cudnn = use_cudnn
    ctx=mx.gpu(0)

    motion_data, class_number, time_step, xyz_rotation =  br.Motion_Data_Preprocessing()
    motion,motion_label=zip(*motion_data)

    '''data loading referenced by Data Loading API '''
    train_iter = mx.io.NDArrayIter(data={'data' : motion},label={'label' : motion_label}, batch_size=batch_size, shuffle=False) #training data
    ####################################################-Network-################################################################
    data = mx.sym.Variable('data')
    label = mx.sym.Variable('label')
    data = mx.sym.transpose(data, axes=(1, 0, 2))  # (time,batch,column)

    #Fused RNN cell results in better performance on GPU(FusedRNNCell supports GPU-only) -> can't be called or stepped!!
    #Fused RNN cell supports Bidirectional , Dropout , multilayer

    if use_cudnn: #faster
        cell = mx.rnn.SequentialRNNCell()
        cell.add(mx.rnn.FusedRNNCell(num_hidden=rnn_hidden_number , num_layers=1 ,bidirectional=False , mode="lstm", prefix="lstm1_", params=None ,forget_bias=1.0 , get_next_state=True))
        cell.add(mx.rnn.DropoutCell(0.3,prefix="lstm_dropout1_"))
        cell.add(mx.rnn.FusedRNNCell(num_hidden=rnn_hidden_number , num_layers=1 ,bidirectional=False , mode="lstm", prefix="lstm2_", params=None ,forget_bias=1.0 , get_next_state=True))
        #Note :  When dropout is set to non-zero in FusedRNNCell, the dropout is applied to the output of all layers except the last layer. If there is only one layer in the FusedRNNCell, the dropout rate is ignored.
    
    else: #general
        cell = mx.rnn.SequentialRNNCell()
        cell.add(mx.rnn.LSTMCell(num_hidden=rnn_hidden_number, prefix="lstm1_"))
        cell.add(mx.rnn.DropoutCell(0.3,prefix="lstm_dropout1_"))
        cell.add(mx.rnn.LSTMCell(num_hidden=rnn_hidden_number, prefix="lstm2_"))

    #if you see the unroll function
    # The hidden state and cell state from the final time step is returned: - state[-1]
    layer, state = cell.unroll(length=time_step, inputs=data, merge_outputs=True, layout='TNC')
    '''FullyConnected Layer'''
    if use_cudnn:
        rnn_output = mx.sym.Reshape(state[-1], shape=(-1, rnn_hidden_number))
        #or
        #rnn_output = mx.sym.Reshape(d_state[-1], shape=(batch_size, rnn_hidden_number)) #using the test_iter, this code don't work, because batch_size is different from train_iter
    else:
        rnn_output = state[-1]

    #if you use dropout
    #rnn_output = mx.sym.Dropout(data=rnn_output,p=0.3)

    affine1 = mx.sym.FullyConnected(data=rnn_output, num_hidden=fc_number, name='affine1') # if use_cudnn=False , data=state[-1] i
    act1 = mx.sym.Activation(data=affine1, act_type='sigmoid', name='sigmoid1')
    drop1 = mx.sym.Dropout(act1, p=0.3)

    affine2 = mx.sym.FullyConnected(data=drop1, num_hidden=class_number, name='affine2')
    output = mx.sym.SoftmaxOutput(data=affine2, label=label, name='softmax')

    # We visualize the network structure with output size (the batch_size is ignored.)
    if use_cudnn:
        shape = {"data": (time_step,batch_size,xyz_rotation)}
        mx.viz.plot_network(symbol=output,shape=shape)#The diagram can be found on the Jupiter notebook.
    else:
        shape = {"data": (batch_size,time_step,xyz_rotation)}
        mx.viz.plot_network(symbol=output,shape=shape)#The diagram can be found on the Jupiter notebook.  

    print(output.list_arguments())

    # training mod
    mod = mx.module.Module(symbol = output , data_names=['data'], label_names=['label'], context=ctx)

    # Network information print
    print(mod.data_names)
    print(mod.label_names)
    print(train_iter.provide_data)
    print(train_iter.provide_label)

    '''if the below code already is declared by mod.fit function, thus we don't have to write it.
    but, when you load the saved weights, you must write the below code.'''
    mod.bind(data_shapes=train_iter.provide_data, label_shapes=train_iter.provide_label)

    # weights save
    if not os.path.exists("weights"):
        os.makedirs("weights")

    if use_cudnn:
        model_name = 'weights/Cudnn_MotionNet'
        checkpoint = mx.callback.do_checkpoint(model_name, period=save_period)
    else:
        model_name = 'weights/MotionNet'
        checkpoint = mx.callback.do_checkpoint(model_name, period=save_period)

    #weights load
    if os.path.exists('weights/Cudnn_MotionNet-1000.params') and use_cudnn==True:
        symbol, arg_params, aux_params = mx.model.load_checkpoint(model_name, 1000)
        mod.set_params(arg_params, aux_params)
    elif os.path.exists('weights/MotionNet-1000.params') and use_cudnn==False:
        symbol, arg_params, aux_params = mx.model.load_checkpoint(model_name, 1000)
        mod.set_params(arg_params, aux_params)

    mod.fit(train_iter, initializer=mx.initializer.Xavier(rnd_type='gaussian', factor_type="avg", magnitude=1),
            optimizer=optimizer,
            optimizer_params={'learning_rate': learning_rate},
            eval_metric=mx.metric.MSE(),
            # Once the loaded parameters are declared here,You do not need to declare mod.set_params,mod.bind
            num_epoch=epoch,
            arg_params=None,
            aux_params=None,
            epoch_end_callback=checkpoint)

    # Network information print
    print(mod.data_shapes)
    print(mod.label_shapes)
    print(mod.output_shapes)
    print(mod.get_params())
    print(mod.get_outputs())
    print("training_data : {}".format(mod.score(train_iter, ['mse', 'acc'])))
    print("Optimization complete.")

    '''test'''
    result = mod.predict(train_iter).asnumpy().argmax(axis=1)
    motion_result=np.argmax(motion_label,axis=1)
    print(result)
    print(motion_result)
    print('Final accuracy : {}%' .format(float(sum(result == motion_result)) / len(result)*100.0))

if __name__ == "__main__":
    print("MotionNet_starting in main")
    MotionNet(epoch=1000 , batch_size=10 , save_period=1000 , optimizer='sgd', learning_rate=0.001 , use_cudnn = True)
else:
    print("MotionNet_imported")