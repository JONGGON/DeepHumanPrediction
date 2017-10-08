import mxnet as mx
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
import bvh_reader as br
import bvh_writer as bw
import decoderRNN as dRNN # JG Customized decoderRNN.py
import encoderRNN as eRNN # JG Customized encoderRNN.py
import os
import time
from collections import OrderedDict
import logging
logging.basicConfig(level=logging.INFO)

'''Let's make my own layer in symbol.'''
#If you want to know more, go to mx.operator.CustomOp.
class LinearRegression(mx.operator.CustomOp):

    '''
    If you want fast speed
    Proceed to mx.ndarray.function !!!
    '''

    def __init__(self, grad_scale):
        #grad_scale -> str
        self.grad_scale = float(grad_scale) #You need to change type to int or float.

    def forward(self, is_train, req, in_data, out_data, aux):
        '''
        in_data[0] -> "input" shape -> (batch size , the number of class)
        in_data[1] -> "label" shape -> (batch size , the number of class)
        out_data[0] -> "output" shape -> (batch size , the number of class)
        '''
        #method1
        out_data[0][:]=in_data[0] # [:]? In python, [:] means copy
        #method2
        """Helper function for assigning into dst depending on requirements."""
        #if necessary
        #self.assign(out_data[0], req[0], in_data[0])

    def backward(self, req, out_grad, in_data, out_data, in_grad, aux):
        '''
        in_data[0] -> "input" shape -> (batch size , the number of class)
        in_data[1] -> "label" shape -> (batch size , the number of class)
        out_data[0] -> "output" shape -> (batch size , the number of class)
        '''

        #method1
        #Mean square Error
        in_grad[0][:] = (out_data[0] - in_data[1])*self.grad_scale # [:]? In python, [:] means copy

        #method2
        """Helper function for assigning into dst depending on requirements."""
        #if necessary
        #self.assign(in_grad[0], req[0], (out_data[0] - in_data[1])*self.grad_scale)


#If you want to know more, go to mx.operator.CustomOpProp.
@mx.operator.register("LinearRegression")
class LinearRegressionProp(mx.operator.CustomOpProp):

    def __init__(self,grad_scale):

        self.grad_scale=grad_scale
        '''
            need_top_grad : bool
        The default declare_backward_dependency function. Use this value
        to determine whether this operator needs gradient input.
        '''
        super(LinearRegressionProp, self).__init__(False)

    #Required.
    def list_arguments(self):
        return ['data', 'label'] # Be sure to write this down. It is a keyword.

    #Required.
    def infer_shape(self, in_shape):
        return [in_shape[0], in_shape[0]], [in_shape[0]]

    #Can be omitted
    def list_outputs(self):
        return ['output']

    #Can be omitted
    def infer_type(self, in_type):
        return in_type, [in_type[0]], []
    '''
    #Define a create_operator function that will be called by the back-end 
    to create an instance of softmax:
    '''
    def create_operator(self, ctx, shapes, dtypes):
        return LinearRegression(self.grad_scale)

print("<<<Motion with advanced Seq2Seq>>>")
'''Encoder'''
def encoder(layer_number=1,hidden_number=500, Dropout_rate=0.2 , Zoneout_rate=0.0 , cell='gru' , parameter_shared=False):

    print("<<<encoder structure>>>")
    cell_type = cell
    param=[]
    Muilti_cell = eRNN.SequentialRNNCell()

    for i in range(layer_number):

        #sharing parameters
        if parameter_shared==True:

            if cell_type == 'gru' or cell_type == 'GRU' or cell_type == 'Gru' :

                if Zoneout_rate > 0:
                    print("zoneoutcell applied-{}".format(i))
                    eGRUCell =eRNN.GRUCell(num_hidden=hidden_number, prefix="{}_{}_shared".format(cell_type, i))
                    param.append(eGRUCell.parameter)
                    Zoneout_cell=eRNN.ZoneoutCell(eGRUCell,zoneout_outputs=Zoneout_rate , zoneout_states=Zoneout_rate)
                    Muilti_cell.add(Zoneout_cell)

                else:
                    eGRUCell =eRNN.GRUCell(num_hidden=hidden_number, prefix="{}_{}_shared".format(cell_type, i))
                    param.append(eGRUCell.parameter)
                    Muilti_cell.add(eGRUCell)

                print("stack {}-'{}'- decoder cell".format(i,cell_type))

            elif cell_type == 'lstm' or cell_type == 'LSTM' or cell_type == 'Lstm' :

                if Zoneout_rate > 0:
                    print("zoneoutcell applied-{}".format(i))
                    eLSTMCell=eRNN.LSTMCell(num_hidden=hidden_number, prefix="{}_{}_shared".format(cell_type, i))
                    param.append(eLSTMCell.parameter)
                    Zoneout_cell=eRNN.ZoneoutCell(eLSTMCell,zoneout_outputs=Zoneout_rate , zoneout_states=Zoneout_rate)
                    Muilti_cell.add(Zoneout_cell)

                else:
                    eLSTMCell=eRNN.LSTMCell(num_hidden=hidden_number, prefix="{}_{}_shared".format(cell_type, i))
                    param.append(eLSTMCell.parameter)
                    Muilti_cell.add(eLSTMCell)

                print("stack {}-'{}'- decoder cell".format(i,cell_type))

            else:

                if Zoneout_rate > 0:
                    print("zoneoutcell applied-{}".format(i))
                    eRNNCell=eRNN.RNNCell(num_hidden=hidden_number, prefix="{}_{}_shared".format(cell_type, i))
                    param.append(eRNNCell.parameter)
                    Zoneout_cell=eRNN.ZoneoutCell(eRNNCell,zoneout_outputs=Zoneout_rate , zoneout_states=Zoneout_rate)
                    Muilti_cell.add(Zoneout_cell)

                else:
                    eRNNCell=eRNN.RNNCell(num_hidden=hidden_number, prefix="{}_{}_shared".format(cell_type, i))
                    param.append(eRNNCell.parameter)
                    Muilti_cell.add(eRNNCell)

                print("stack {}-'{}'- encoder cell".format(i,cell_type))

            if Dropout_rate > 0 and (layer_number - 1) > i:

                Muilti_cell.add(eRNN.DropoutCell(Dropout_rate, prefix="dropout_encoder_{}".format(i)))
                print("stack {}-'{}'- encoder dropout cell".format(i,cell_type))

        else : #parameter_shared==False:
            if cell_type == 'gru' or cell_type == 'GRU' or cell_type == 'Gru' :

                if Zoneout_rate > 0:
                    print("zoneoutcell applied-{}".format(i))
                    Zoneout_cell=eRNN.ZoneoutCell(eRNN.GRUCell(num_hidden=hidden_number , prefix="{}_encoder_{}".format(cell_type,i)),zoneout_outputs=Zoneout_rate , zoneout_states=Zoneout_rate)
                    Muilti_cell.add(Zoneout_cell)

                else:
                    Muilti_cell.add(eRNN.GRUCell(num_hidden=hidden_number , prefix="{}_encoder_{}".format(cell_type,i)))

                print("stack {}-'{}'- decoder cell".format(i,cell_type))

            elif cell_type == 'lstm' or cell_type == 'LSTM' or cell_type == 'Lstm' :


                if Zoneout_rate > 0:
                    print("zoneoutcell applied-{}".format(i))
                    Zoneout_cell=eRNN.ZoneoutCell(eRNN.LSTMCell(num_hidden=hidden_number, prefix="{}_encoder_{}".format(cell_type,i)),zoneout_outputs=Zoneout_rate , zoneout_states=Zoneout_rate)
                    Muilti_cell.add(Zoneout_cell)

                else:
                    Muilti_cell.add(eRNN.LSTMCell(num_hidden=hidden_number,prefix="{}_encoder_{}".format(cell_type,i)))

                print("stack {}-'{}'- decoder cell".format(i,cell_type))

            else:

                if Zoneout_rate > 0:
                    print("zoneoutcell applied-{}".format(i))
                    Zoneout_cell=eRNN.ZoneoutCell(eRNN.RNNCell(num_hidden=hidden_number, prefix="{}_encoder_{}".format(cell_type,i)),zoneout_outputs=Zoneout_rate , zoneout_states=Zoneout_rate)
                    Muilti_cell.add(Zoneout_cell)

                else:
                    Muilti_cell.add(eRNN.RNNCell(num_hidden=hidden_number, prefix="{}_encoder_{}".format(cell_type,i)))

                print("stack {}-'{}'- encoder cell".format(i,cell_type))

            if Dropout_rate > 0 and (layer_number - 1) > i:

                Muilti_cell.add(eRNN.DropoutCell(Dropout_rate, prefix="dropout_encoder_{}".format(i)))
                print("stack {}-'{}'- encoder dropout cell".format(i,cell_type))


    print("\n")
    return Muilti_cell,param

'''Decoder'''
def decoder(layer_number=1 , hidden_number=500 , output_number = 100 , Dropout_rate=0.2 , Zoneout_rate=0.0 , Residual=True , cell='gru', param=list()):

    print("<<<decoder structure>>>")

    cell_type = cell
    Muilti_cell = dRNN.SequentialRNNCell()

    for i in range(layer_number):
        #sharing parameters
        if param!=list():

            if cell_type == 'gru' or cell_type == 'GRU' or cell_type == 'Gru' :

                if Residual == True and Zoneout_rate > 0:
                    print("residualcell applied_{}".format(i))
                    Residual_cell=dRNN.ResidualCell(dRNN.GRUCell(current_layer=i ,layer_number=layer_number,num_hidden=hidden_number, num_output= output_number ,prefix="{}_{}_shared".format(cell_type,i) ,shared_param=param[i]))
                    print("zoneoutcell applied-{}".format(i))
                    Zoneout_cell=dRNN.ZoneoutCell(Residual_cell,zoneout_outputs=Zoneout_rate , zoneout_states=Zoneout_rate)
                    Muilti_cell.add(Zoneout_cell)

                elif Residual == True and Zoneout_rate == 0:
                    print("residualcell applied_{}".format(i))
                    Residual_cell=dRNN.ResidualCell(dRNN.GRUCell(current_layer=i ,layer_number=layer_number,num_hidden=hidden_number, num_output= output_number ,prefix="{}_{}_shared".format(cell_type,i),shared_param=param[i]))
                    Muilti_cell.add(Residual_cell)

                elif Residual == False and Zoneout_rate > 0:
                    print("zoneoutcell applied-{}".format(i))
                    Zoneout_cell=dRNN.ZoneoutCell(dRNN.GRUCell(current_layer=i ,layer_number=layer_number , num_hidden=hidden_number, num_output= output_number ,prefix="{}_{}_shared".format(cell_type,i),shared_param=param[i]) , zoneout_outputs=Zoneout_rate , zoneout_states=Zoneout_rate)
                    Muilti_cell.add(Zoneout_cell)

                else:
                    Muilti_cell.add(dRNN.GRUCell(current_layer=i , layer_number=layer_number,num_hidden=hidden_number, num_output= output_number ,prefix="{}_{}_shared".format(cell_type,i),shared_param=param[i]))

                print("stack {}-'{}'- decoder cell".format(i,cell_type))

            elif cell_type == 'lstm' or cell_type == 'LSTM' or cell_type == 'Lstm':

                if Residual == True and Zoneout_rate > 0:
                    print("residualcell applied_{}".format(i))
                    Residual_cell=dRNN.ResidualCell(dRNN.LSTMCell(current_layer=i ,layer_number=layer_number,num_hidden=hidden_number, num_output=output_number, prefix="{}_{}_shared".format(cell_type,i),shared_param=param[i]))
                    print("zoneoutcell applied-{}".format(i))
                    Zoneout_cell=dRNN.ZoneoutCell(Residual_cell,zoneout_outputs=Zoneout_rate , zoneout_states=Zoneout_rate)
                    Muilti_cell.add(Zoneout_cell)

                elif Residual == True and Zoneout_rate == 0:
                    print("residualcell applied_{}".format(i))
                    Residual_cell=dRNN.ResidualCell(dRNN.LSTMCell(num_hidden=hidden_number, num_output=output_number, current_layer=i , layer_number=layer_number, prefix="{}_{}_shared".format(cell_type,i),shared_param=param[i]))
                    Muilti_cell.add(Residual_cell)

                elif Residual == False and Zoneout_rate > 0:
                    print("zoneoutcell applied-{}".format(i))
                    Zoneout_cell=dRNN.ZoneoutCell(dRNN.LSTMCell(current_layer=i ,layer_number=layer_number,num_hidden=hidden_number, num_output=output_number, prefix="{}_{}_shared".format(cell_type,i),shared_param=param[i]),zoneout_outputs=Zoneout_rate , zoneout_states=Zoneout_rate)
                    Muilti_cell.add(Zoneout_cell)

                else:
                    Muilti_cell.add(dRNN.LSTMCell(current_layer=i , layer_number=layer_number , num_hidden=hidden_number , num_output=output_number , prefix="{}_{}_shared".format(cell_type,i),shared_param=param[i]))

                print("stack {}-'{}'- decoder cell".format(i,cell_type))

            else:

                if Residual == True and Zoneout_rate > 0:
                    print("residualcell applied_{}".format(i))
                    Residual_cell=dRNN.ResidualCell(dRNN.RNNCell(current_layer=i , layer_number=layer_number , num_hidden=hidden_number, num_output=output_number ,prefix="{}_{}_shared".format(cell_type,i),shared_param=param[i]))
                    print("zoneoutcell applied-{}".format(i))
                    Zoneout_cell=dRNN.ZoneoutCell(Residual_cell,zoneout_outputs=Zoneout_rate , zoneout_states=Zoneout_rate)
                    Muilti_cell.add(Zoneout_cell)

                elif Residual == True and Zoneout_rate == 0:
                    print("residualcell applied_{}".format(i))
                    Residual_cell=dRNN.ResidualCell(dRNN.RNNCell(current_layer=i , layer_number=layer_number , num_hidden=hidden_number, num_output=output_number, prefix="{}_{}_shared".format(cell_type,i),shared_param=param[i]))
                    Muilti_cell.add(Residual_cell)

                elif Residual == False and Zoneout_rate > 0:
                    print("zoneoutcell applied-{}".format(i))
                    Zoneout_cell=dRNN.ZoneoutCell(dRNN.RNNCell(current_layer=i , layer_number=layer_number , num_hidden=hidden_number, num_output=output_number , prefix="{}_{}_shared".format(cell_type,i),shared_param=param[i]),zoneout_outputs=Zoneout_rate , zoneout_states=Zoneout_rate)
                    Muilti_cell.add(Zoneout_cell)

                else:
                    Muilti_cell.add(dRNN.RNNCell(current_layer=i , layer_number=layer_number , num_hidden=hidden_number , num_output=output_number, prefix="{}_{}_shared".format(cell_type,i),shared_param=param[i]))

                print("stack {}-'{}'- decoder cell".format(i,cell_type))

            if Dropout_rate > 0 and (layer_number-1) > i:
                Muilti_cell.add(dRNN.DropoutCell(Dropout_rate, prefix="dropout_decoder_{}".format(i)))
                print("stack {}-'{}'- decoder dropout cell".format(i,cell_type))

        else: #param = list() or []:
            if cell_type == 'gru' or cell_type == 'GRU' or cell_type == 'Gru' :

                if Residual == True and Zoneout_rate > 0:
                    print("residualcell applied_{}".format(i))
                    Residual_cell=dRNN.ResidualCell(dRNN.GRUCell(current_layer=i ,layer_number=layer_number , num_hidden=hidden_number, num_output= output_number ,prefix="{}_decoder_{}".format(cell_type,i)))
                    print("zoneoutcell applied-{}".format(i))
                    Zoneout_cell=dRNN.ZoneoutCell(Residual_cell,zoneout_outputs=Zoneout_rate , zoneout_states=Zoneout_rate)
                    Muilti_cell.add(Zoneout_cell)

                elif Residual == True and Zoneout_rate == 0:
                    print("residualcell applied_{}".format(i))
                    Residual_cell=dRNN.ResidualCell(dRNN.GRUCell(current_layer=i ,layer_number=layer_number , num_hidden=hidden_number, num_output= output_number ,prefix="{}_decoder_{}".format(cell_type,i)))
                    Muilti_cell.add(Residual_cell)

                elif Residual == False and Zoneout_rate > 0:
                    print("zoneoutcell applied-{}".format(i))
                    Zoneout_cell=dRNN.ZoneoutCell(dRNN.GRUCell(current_layer=i ,layer_number=layer_number , num_hidden=hidden_number, num_output= output_number ,prefix="{}_decoder_{}".format(cell_type,i)),zoneout_outputs=Zoneout_rate , zoneout_states=Zoneout_rate)
                    Muilti_cell.add(Zoneout_cell)

                else:
                    Muilti_cell.add(dRNN.GRUCell(current_layer=i ,layer_number=layer_number , num_hidden=hidden_number, num_output= output_number ,prefix="{}_decoder_{}".format(cell_type,i)))

                print("stack {}-'{}'- decoder cell".format(i,cell_type))

            elif cell_type == 'lstm' or cell_type == 'LSTM' or cell_type == 'Lstm':

                if Residual == True and Zoneout_rate > 0:
                    print("residualcell applied_{}".format(i))
                    Residual_cell=dRNN.ResidualCell(dRNN.LSTMCell(current_layer=i ,layer_number=layer_number , num_hidden=hidden_number, num_output=output_number, prefix="{}_decoder_{}".format(cell_type,i)))
                    print("zoneoutcell applied-{}".format(i))
                    Zoneout_cell=dRNN.ZoneoutCell(Residual_cell,zoneout_outputs=Zoneout_rate , zoneout_states=Zoneout_rate)
                    Muilti_cell.add(Zoneout_cell)

                elif Residual == True and Zoneout_rate == 0:
                    print("residualcell applied_{}".format(i))
                    Residual_cell=dRNN.ResidualCell(dRNN.LSTMCell(current_layer=i ,layer_number=layer_number , num_hidden=hidden_number, num_output=output_number, prefix="{}_decoder_{}".format(cell_type,i)))
                    Muilti_cell.add(Residual_cell)

                elif Residual == False and Zoneout_rate > 0:
                    print("zoneoutcell applied-{}".format(i))
                    Zoneout_cell=dRNN.ZoneoutCell(dRNN.LSTMCell(current_layer=i ,layer_number=layer_number , num_hidden=hidden_number, num_output=output_number, prefix="{}_decoder_{}".format(cell_type,i)),zoneout_outputs=Zoneout_rate , zoneout_states=Zoneout_rate)
                    Muilti_cell.add(Zoneout_cell)

                else:
                    Muilti_cell.add(dRNN.LSTMCell(current_layer=i ,layer_number=layer_number , num_hidden=hidden_number, num_output=output_number,prefix="{}_decoder_{}".format(cell_type,i)))

                print("stack {}-'{}'- decoder cell".format(i,cell_type))

            else:

                if Residual == True and Zoneout_rate > 0:
                    print("residualcell applied_{}".format(i))
                    Residual_cell=dRNN.ResidualCell(dRNN.RNNCell(current_layer=i ,layer_number=layer_number , num_hidden=hidden_number, num_output=output_number , prefix="{}_decoder_{}".format(cell_type,i)))
                    print("zoneoutcell applied-{}".format(i))
                    Zoneout_cell=dRNN.ZoneoutCell(Residual_cell,zoneout_outputs=Zoneout_rate , zoneout_states=Zoneout_rate)
                    Muilti_cell.add(Zoneout_cell)

                elif Residual == True and Zoneout_rate == 0:
                    print("residualcell applied_{}".format(i))
                    Residual_cell=dRNN.ResidualCell(dRNN.RNNCell(current_layer=i ,layer_number=layer_number , num_hidden=hidden_number, num_output=output_number,  prefix="{}_decoder_{}".format(cell_type,i)))
                    Muilti_cell.add(Residual_cell)

                elif Residual == False and Zoneout_rate > 0:
                    print("zoneoutcell applied-{}".format(i))
                    Zoneout_cell=dRNN.ZoneoutCell(dRNN.RNNCell(num_hidden=hidden_number, num_output=output_number , prefix="{}_decoder_{}".format(cell_type,i)),zoneout_outputs=Zoneout_rate , zoneout_states=Zoneout_rate)
                    Muilti_cell.add(Zoneout_cell)

                else:
                    Muilti_cell.add(dRNN.RNNCell(current_layer=i ,layer_number=layer_number , num_hidden=hidden_number , num_output=output_number, prefix="{}_decoder_{}".format(cell_type,i)))

                print("stack {}-'{}'- decoder cell".format(i,cell_type))

            if Dropout_rate > 0 and (layer_number-1) > i:
                Muilti_cell.add(dRNN.DropoutCell(Dropout_rate, prefix="dropout_decoder_{}".format(i)))
                print("stack {}-'{}'- decoder dropout cell".format(i,cell_type))

    print("\n")
    return Muilti_cell

def MotionNet(epoch=None , batch_size=None , save_period=None , cost_limit=None ,
    optimizer=None, learning_rate=None , lr_step=None , lr_factor=None , stop_factor_lr=None , use_gpu=True ,
    TEST=None , num_layer=None , cell=None, hidden_unit=None ,time_step = None , seed_timestep = None , batch_Frame= None , frame_time=None, graphviz=None , parameter_shared=True , Model = None):

    print("-------------------Motion Net-------------------")

    '''1. Data_Loading - bvh_reader'''
    Normalization_factor, train_motion, train_label_motion , seed_timestep, pre_timestep, column, file_directory = br.Motion_Data_Preprocessing(time_step , seed_timestep , batch_Frame , TEST , Model)

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
        pre_motion = mx.sym.reshape(mx.sym.slice_axis(data=all_motion, axis=1, begin=seed_timestep-1, end=seed_timestep),
                                shape=(1, -1))  # (batch=1,column) - first frame
    else:
        pre_motion = mx.sym.reshape(mx.sym.slice_axis(data=all_motion, axis=1, begin=seed_timestep-1, end=seed_timestep),
                                shape=(batch_size, -1))  # (batch,column) - first frame

    print("-------------------Network Shape--------------------")
    '''
    only if encoder's parameter_shared=True , parameter Encoder and decoder parameters are shared.
    '''
    e_cell , encoder_parameter = encoder(layer_number= num_layer , hidden_number=hidden_unit , Dropout_rate=0.0 , Zoneout_rate=0.0 , cell=cell , parameter_shared = parameter_shared ) # if parameter_shared = True, paramter = encoder's weights , else False = []

    if num_layer==1 and parameter_shared==True: #only if num_layer=1 , Both Residual = True and Residual = False are possible.
        d_cell = decoder(layer_number= num_layer , hidden_number=hidden_unit , output_number=column , Dropout_rate=0.0 , Zoneout_rate=0.0 , Residual = True , cell=cell , param=encoder_parameter)

    elif num_layer>1 and parameter_shared==True:# There is no way to share parameters with the encoder.
        d_cell = decoder(layer_number=num_layer, hidden_number=hidden_unit, output_number=column, Dropout_rate=0.0,Zoneout_rate=0.0, Residual=False, cell=cell, param=encoder_parameter)

    else: # parameter_shared=False , Both Residual = True and Residual = False are possible.
        d_cell = decoder(layer_number=num_layer, hidden_number=hidden_unit, output_number=column, Dropout_rate=0.0,Zoneout_rate=0.0, Residual=True, cell=cell, param=encoder_parameter)
    print("\n")

    _ , e_state = e_cell.unroll(length=seed_timestep , inputs=seed_motion , merge_outputs=True , layout='NTC')

    # customized by JG
    if num_layer==1:
        d_output, _ = d_cell.SingleLayer_feed_previous_unroll(length=pre_timestep, begin_state=e_state,inputs=pre_motion, merge_outputs=True, layout='NTC') # MultiLayer_feed_previous_unroll is also possible.
    else:
        d_output, _ = d_cell.MultiLayer_feed_previous_unroll(length=pre_timestep, begin_state=e_state, inputs=pre_motion, merge_outputs=True, layout='NTC') # MultiLayer_feed_previous_unroll is also possible.


    #output = mx.sym.LinearRegressionOutput(data = d_output , label=label_motion , grad_scale = 1)
    output = mx.sym.Custom(data=d_output, label=label_motion, grad_scale=1, name="LinearRegression", op_type='LinearRegression')

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

                if epoch % 2 == 0: # Only noise is added when it is an even number
                    '''1. Add noise to input - Data Augmentation'''
                    #random_normal
                    noise = mx.nd.random_normal(loc=0 , scale=5 , shape=(batch_size , seed_timestep+pre_timestep , column) , ctx=ctx) # random_normal
                    #random_uniform
                    #noise = mx.nd.random_uniform(low=-1 , high=1 , shape=(batch_size , seed_timestep+pre_timestep , column) ,ctx=ctx) # random_uniform
                    mod.forward(data_batch=mx.io.DataBatch(data = list([mx.nd.add(batch.data[0].as_in_context(ctx), noise)]), label= list(batch.label)), is_train=True)
                    #Data Order Transform (N,T,C)
                else:
                    mod.forward(data_batch=batch , is_train=True)

                mod.update_metric(metric,batch.label)
                mod.backward()
                mod.update()

            #print('epoch : {} , MSE : {}'.format(epoch,metric.get()))
            if epoch % 100 == 0:
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

            print(style.available)
            TimeStepError_Array=np.mean(cost,axis=(0,2)) # y-axis
            print(TimeStepError_Array)
            TimeStep = np.arange(1,pre_timestep+1,1) # x-axis

            #Dram Error graph
            style.use('seaborn')
            plt.figure(figsize=(9,4))
            bbox = dict(boxstyle = 'round' , fc = 'w' , ec = 'b' , lw = 2)
            #plt.plot(TimeStep , TimeStepError_Array , "r." , lw=3 ,label = "Error")
            plt.bar(TimeStep , TimeStepError_Array , width=0.7 ,label ='error', color = 'r')
            plt.annotate("Error Prevention" ,fontsize=14, xy = (60,1000) , xytext=(0,1000), textcoords='data' ,arrowprops={'color' : 'blue' , 'alpha' : 0.3 , 'arrowstyle' : "simple"}, bbox = bbox)
            plt.grid()
            plt.xlabel("Time", fontsize=14)
            plt.ylabel("Joint Angle Error" , fontsize=14)
            plt.ylim(0,4400)
            plt.legend(fontsize=15,loc='upper left')
            plt.title("Prediction Error Graph", fontdict={'fontsize': 15 , 'fontweight' : 5})
            print("cost graph saved")
            plt.savefig("Cost Graph.jpg")

            cost=np.mean(cost)
            print("prediction error : {}".format(cost))

            '''Creating a bvh file with predicted values -bvh_writer'''
            bw.Motion_Data_Making(seed[:,:seed_timestep] / Normalization_factor , prediction_motion , seed_timestep , pre_timestep , batch_Frame , frame_time , file_directory , Model)

            plt.show()
            return "Test completed"
        else:
            print("Can not test")


if __name__ == "__main__":

    TEST=False
    Model=3

    #The following parameters must have the same value in 'training' and 'test' modes.
    num_layer=1  
    cell='lstm'
    hidden_unit=1000  
    time_step = 90
    seed_timestep = 30
    batch_Frame= 1  
    frame_time = 30
    save_period=0
    parameter_shared = True

    '''Execution'''
    if TEST : 

        completed=MotionNet(TEST=TEST , save_period=1 , num_layer=num_layer , cell=cell, hidden_unit=hidden_unit , time_step = time_step , seed_timestep = seed_timestep , batch_Frame= batch_Frame , frame_time=frame_time ,graphviz=True, parameter_shared=parameter_shared, Model=Model)
        print(completed)

    else:
        completed = MotionNet(epoch=700000 , batch_size=68 , save_period=save_period, cost_limit=0.1 ,
        optimizer='adam', learning_rate=0.0001 , lr_step=5000, lr_factor=0.99, stop_factor_lr=1e-08 , use_gpu=True ,
        TEST=TEST , num_layer=num_layer , cell=cell , hidden_unit=hidden_unit , time_step = time_step , seed_timestep = seed_timestep , batch_Frame = batch_Frame , frame_time=frame_time , graphviz=True , parameter_shared=parameter_shared)
        print(completed)

else:
    print("MotionNet_imported")