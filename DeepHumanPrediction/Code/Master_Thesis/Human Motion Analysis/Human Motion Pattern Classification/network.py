import mxnet as mx
import bvh_reader as br
import os
import time
from collections import OrderedDict
import logging
logging.basicConfig(level=logging.INFO)

print("Motion Classification")

def MotionNet(epoch=None , batch_size=None , save_period=None ,optimizer=None, learning_rate=None , Dropout=0.0 , use_gpu=True , use_cudnn=True , TEST=None , Model=None,  num_layer=None , hidden_unit=None ,time_step = None , batch_Frame= None , graphviz=None ):

    print("-------------------Motion Net-------------------")
    if use_gpu:
        ctx = mx.gpu(0)
    else:
        ctx = mx.cpu(0)

    '''1. Data_Loading - bvh_reader'''
    Normalization_factor, motion_data, motion_label , time_step , class_number ,column = br.Motion_Data_Preprocessing(time_step , batch_Frame , TEST , Model)
    motion_one_hot_label = mx.nd.one_hot(mx.nd.array(motion_label,ctx), class_number)

    if TEST==True:
        print("<Test>")
        data = OrderedDict()
        data['motion'] = motion_data
        label = {'label': motion_label}

        test_iter = mx.io.NDArrayIter(data=data, label=label)

    else:
        print("<Training>")
        data = OrderedDict()
        data['motion'] = motion_data
        label = {'label': motion_label}

        train_iter = mx.io.NDArrayIter(data=data, label=label, batch_size=batch_size, shuffle=False, last_batch_handle='pad')  # Motion data is complex and requires sequential learning to learn from easy examples. So shuffle = False ->In here, not using sequential learning

    '''2. Network'''
    motion = mx.sym.Variable('motion') #(batch , time ,column)
    label = mx.sym.Variable('label')
    motion = mx.sym.transpose(motion , axes=(1,0,2)) # (time , batch , column)

    cell = mx.rnn.SequentialRNNCell()
    if use_cudnn:  # even faster -  real!!!
        for i in range(num_layer):
            #When dropout is set to non-zero in FusedRNNCell, the dropout is applied to the output of all layers except the last layer. If there is only one layer in the FusedRNNCell, the dropout rate is ignored.
            cell.add(mx.rnn.FusedRNNCell(num_hidden=hidden_unit, num_layers=1, bidirectional=False, mode="lstm", prefix="lstm_{}".format(i), params=None, forget_bias=1.0, get_next_state=False))
            cell.add(mx.rnn.DropoutCell(Dropout, prefix="lstm_dropout{}_".format(i)))
    else:  # general
        for i in range(num_layer):
            cell.add(mx.rnn.LSTMCell(num_hidden=hidden_unit, prefix="lstm{}_".format(i)))
            cell.add(mx.rnn.DropoutCell(Dropout, prefix="lstm_dropout{}_".format(i)))

    print("-------------------Network Shape--------------------")
    output , _ = cell.unroll(length=time_step , inputs = motion , merge_outputs=False , layout='TNC')

    '''FullyConnected Layer'''
    #if use_cudnn:
    #    output = mx.sym.Reshape(output[-1], shape=(-1, hidden_unit))
    #else:
    #    output = output[-1]

    affine1 = mx.sym.FullyConnected(data=output[-1], num_hidden=hidden_unit, name='affine1') # if use_cudnn=False , data=state[-1] i
    act1 = mx.sym.Activation(data=affine1, act_type='relu', name='relu')
    dropout1 = mx.sym.Dropout(act1, p=Dropout)
    affine2 = mx.sym.FullyConnected(data=dropout1, num_hidden=class_number, name='affine2')
    output = mx.sym.SoftmaxOutput(data=affine2, label=label  , grad_scale=1 , name="softmaxoutput")
    digraph=mx.viz.plot_network(symbol=output,hide_weights=True)

    if graphviz==True:
        digraph.view("Network Structure") #show graph

    print("-------------------Network Learning Parameter--------------------")
    print(output.list_arguments())
    print("\n")

    if TEST == False:
        mod = mx.module.Module(symbol=output, data_names=['motion'], label_names=['label'], context=ctx)
        print("-------------------Network Data Name--------------------")
        print(mod.data_names)
        print(mod.label_names)
        print("\n")
    else:
        # test mod
        test_mod = mx.mod.Module(symbol=output , data_names=['motion'] , label_names=['label'] , context=ctx)
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


    if TEST == False:

        mod.bind(data_shapes=train_iter.provide_data , label_shapes=train_iter.provide_label , for_training=True , shared_module=None , inputs_need_grad=False , force_rebind=False , grad_req='write')
        # weights load
        if use_cudnn:
            weights_path = 'weights/cudnn_MotionNet-{}.params'.format(save_period)
        else:
            weights_path = 'weights/MotionNet-{}.params'.format(save_period)

        if os.path.exists(weights_path):
            print("Load weights")
            mod.load_params(weights_path)
        else:
            print("Init weights")
            mod.init_params(initializer=mx.initializer.Normal(sigma=0.01)) # very important

        start_time=time.time()
        print("-------------------Learning Start--------------------")

        mod.init_optimizer(optimizer=optimizer, optimizer_params={'learning_rate': learning_rate})

        for epoch in range(1, epoch + 1, 1):
            train_iter.reset()
            for batch in train_iter:

                if epoch % 2 == 0: # Only noise is added when it is an even number
                    '''1. Add noise to input - Data Augmentation'''
                    #random_normal
                    noise = mx.nd.random_normal(loc=0 , scale=1 , shape=(batch_size , time_step , column) , ctx=ctx) # random_normal
                    #random_uniform
                    #noise = mx.nd.random_uniform(low=-1 , high=1 , shape=(batch_size , time_step , column) ,ctx=ctx) # random_uniform
                    mod.forward(data_batch=mx.io.DataBatch(data = list([mx.nd.add(batch.data[0].as_in_context(ctx), noise)]), label= list(batch.label)), is_train=True)
                else:
                    mod.forward(data_batch=batch , is_train=True)

                mod.backward()
                mod.update()

            if not os.path.exists("weights"):
                os.makedirs("weights")

            if epoch % 10000 == 0:
                end_time=time.time()
                print("-------------------------------------------------------")
                print("{}_learning time : {}".format(epoch,end_time-start_time))
                print("-------------------------------------------------------")

            if epoch % 10000 == 0 and use_cudnn==True:
                print('Saving weights')
                mod.save_params("weights/cudnn_MotionNet-{}.params".format(epoch))
            elif  epoch % 10000 == 0 and use_cudnn==False:
                print('Saving weights')
                mod.save_params("weights/MotionNet-{}.params".format(epoch))

            motion_class = mod.predict(eval_data=train_iter ,  merge_batches=True , reset=True, always_output_list=False) / Normalization_factor
            cost=motion_class-motion_one_hot_label
            cost=mx.nd.divide(mx.nd.square(cost),2)
            cost=mx.nd.mean(cost)
            print("epoch : {} , ".format(epoch) , end="")
            print("cost : {}".format(cost.asscalar()))
            print('Training Accuracy : {0:0.4f}%'.format(float(sum(motion_class.asnumpy().argmax(axis=1) == motion_label)) / len(motion_label) * 100.0))

        if use_cudnn == True:
            print('Saving Final weights')
            mod.save_params("weights/cudnn_MotionNet-{}.params".format(epoch))
        else:
            print('Saving Final weights')
            mod.save_params("weights/MotionNet-{}.params".format(epoch))

        print("\n")
        print("-------------------Network Information--------------------")
        print(mod.data_shapes)
        print(mod.label_shapes)
        print(mod.output_shapes)
        #print(mod.get_params())
        #print(mod.get_outputs())
        print("{} learning optimization completed".format(epoch))
        print("\n")

    # not percentage!!!
    if TEST==True:

        test_mod.bind(data_shapes=test_iter.provide_data , label_shapes=test_iter.provide_label , for_training=False)

        # weights load
        if use_cudnn:
            weights_path = 'weights/cudnn_MotionNet-{}.params'.format(save_period)
        else:
            weights_path = 'weights/MotionNet-{}.params'.format(save_period)

        if os.path.exists(weights_path):
            test_mod.load_params(weights_path)

            motion_class = test_mod.predict(eval_data=test_iter , merge_batches=True , always_output_list=False)/Normalization_factor
            print(motion_class.asnumpy().argmax(axis=1))
            #print('Test Accuracy : {}%'.format(float(sum(motion_class.asnumpy().argmax(axis=1) == motion_label)) / len(motion_label) * 100.0))

            print("Test completed")
        else:
            print("Can not test")


if __name__ == "__main__":

    TEST = False

    # The following parameters must have the same value in 'training' and 'test' modes.
    num_layer = 1
    hidden_unit = 1000
    time_step = 90
    batch_Frame = 1
    save_period = 10000
    use_gpu = True
    use_cudnn = True
    Model = 1  # # Only 1, 2, and 3 are possible and only works when TEST = True.
    '''Execution'''
    if TEST:
        MotionNet(TEST=TEST, Model=1, save_period=save_period, num_layer=num_layer, hidden_unit=hidden_unit,
                  time_step=time_step, batch_Frame=batch_Frame, use_gpu=use_gpu, use_cudnn=use_cudnn, graphviz=False)
    else:
        # batch learning
        MotionNet(epoch=5000, batch_size=68, save_period=save_period, optimizer='adam', learning_rate=0.01, Dropout=0.2,
                  use_gpu=use_gpu, use_cudnn=use_cudnn,
                  TEST=TEST, num_layer=num_layer, hidden_unit=hidden_unit, time_step=time_step, batch_Frame=batch_Frame,
                  graphviz=False)

else:
    print("MotionNet_imported")