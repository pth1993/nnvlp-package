import lasagne
import theano.tensor as T
import theano
from lasagne.layers import Gate
import lasagne.nonlinearities as nonlinearities
from . import utils
from lasagne import init
from lasagne.layers import MergeLayer


class CRFLayer(MergeLayer):
    def __init__(self, incoming, num_labels, mask_input=None, W=init.GlorotUniform(), b=init.Constant(0.), **kwargs):
        self.input_shape = incoming.output_shape
        incomings = [incoming]
        self.mask_incoming_index = -1
        if mask_input is not None:
            incomings.append(mask_input)
            self.mask_incoming_index = 1

        super(CRFLayer, self).__init__(incomings, **kwargs)
        self.num_labels = num_labels + 1
        self.pad_label_index = num_labels

        num_inputs = self.input_shape[2]
        self.W = self.add_param(W, (num_inputs, self.num_labels, self.num_labels), name="W")

        if b is None:
            self.b = None
        else:
            self.b = self.add_param(b, (self.num_labels, self.num_labels), name="b", regularizable=False)

    def get_output_shape_for(self, input_shapes):
        input_shape = input_shapes[0]
        return input_shape[0], input_shape[1], self.num_labels, self.num_labels

    def get_output_for(self, inputs, **kwargs):
        input = inputs[0]
        mask = None
        if self.mask_incoming_index > 0:
            mask = inputs[self.mask_incoming_index]

        # compute out by tensor dot ([batch, length, input] * [input, num_label, num_label]
        # the shape of out should be [batch, length, num_label, num_label]
        out = T.tensordot(input, self.W, axes=[[2], [0]])

        if self.b is not None:
            b_shuffled = self.b.dimshuffle('x', 'x', 0, 1)
            out = out + b_shuffled

        if mask is not None:
            mask_shuffled = mask.dimshuffle(0, 1, 'x', 'x')
            out = out * mask_shuffled
        return out


def build_model(embedd_dim, max_sent_length, max_char_length, char_alphabet_size, char_embedd_dim, num_labels, dropout,
                num_filters, num_units, grad_clipping, peepholes, char_embedd_table):
    # create target layer
    target_var = T.imatrix(name='targets')
    # create mask layer
    mask_var = T.matrix(name='masks', dtype=theano.config.floatX)
    layer_mask = lasagne.layers.InputLayer(shape=(None, max_sent_length), input_var=mask_var, name='mask')
    # create word input and char input layers
    input_var = T.tensor3(name='inputs', dtype=theano.config.floatX)
    char_input_var = T.itensor3(name='char-inputs')
    layer_embedding = lasagne.layers.InputLayer(shape=(None, max_sent_length, embedd_dim), input_var=input_var,
                                                name='input')
    layer_char_input = lasagne.layers.InputLayer(shape=(None, max_sent_length, max_char_length),
                                                 input_var=char_input_var, name='char-input')
    layer_char_input = lasagne.layers.reshape(layer_char_input, (-1, [2]))
    layer_char_embedding = lasagne.layers.EmbeddingLayer(layer_char_input, input_size=char_alphabet_size,
                                                         output_size=char_embedd_dim, name='char_embedding',
                                                         W=char_embedd_table)
    layer_char_input = lasagne.layers.DimshuffleLayer(layer_char_embedding, pattern=(0, 2, 1))
    # create cnn
    conv_window = 3
    _, sent_length, _ = layer_embedding.output_shape
    # dropout before cnn?
    if dropout:
        layer_char_input = lasagne.layers.DropoutLayer(layer_char_input, p=0.5)
    # construct convolution layer
    cnn_layer = lasagne.layers.Conv1DLayer(layer_char_input, num_filters=num_filters,
                                           filter_size=conv_window, pad='full',
                                           nonlinearity=lasagne.nonlinearities.tanh, name='cnn')
    # infer the pool size for pooling (pool size should go through all time step of cnn)
    _, _, pool_size = cnn_layer.output_shape
    # construct max pool layer
    pool_layer = lasagne.layers.MaxPool1DLayer(cnn_layer, pool_size=pool_size)
    # reshape the layer to match lstm incoming layer [batch * sent_length, num_filters, 1] --> [batch, sent_length,
    # num_filters]
    output_cnn_layer = lasagne.layers.reshape(pool_layer, (-1, sent_length, [1]))
    # finally, concatenate the two incoming layers together.
    incoming = lasagne.layers.concat([output_cnn_layer, layer_embedding], axis=2)
    # create bi-lstm
    if dropout:
        incoming = lasagne.layers.DropoutLayer(incoming, p=0.5)
    ingate_forward = Gate(W_in=lasagne.init.GlorotUniform(), W_hid=lasagne.init.GlorotUniform(),
                          W_cell=lasagne.init.Uniform(range=0.1))
    outgate_forward = Gate(W_in=lasagne.init.GlorotUniform(), W_hid=lasagne.init.GlorotUniform(),
                           W_cell=lasagne.init.Uniform(range=0.1))
    # according to Jozefowicz et al.(2015), init bias of forget gate to 1.
    forgetgate_forward = Gate(W_in=lasagne.init.GlorotUniform(), W_hid=lasagne.init.GlorotUniform(),
                              W_cell=lasagne.init.Uniform(range=0.1), b=lasagne.init.Constant(1.))
    # now use tanh for nonlinear function of cell, need to try pure linear cell
    cell_forward = Gate(W_in=lasagne.init.GlorotUniform(), W_hid=lasagne.init.GlorotUniform(), W_cell=None,
                        nonlinearity=nonlinearities.tanh)
    lstm_forward = lasagne.layers.LSTMLayer(incoming, num_units, mask_input=layer_mask,
                                            grad_clipping=grad_clipping, nonlinearity=nonlinearities.tanh,
                                            peepholes=peepholes, ingate=ingate_forward, outgate=outgate_forward,
                                            forgetgate=forgetgate_forward, cell=cell_forward, name='forward')
    ingate_backward = Gate(W_in=lasagne.init.GlorotUniform(), W_hid=lasagne.init.GlorotUniform(),
                           W_cell=lasagne.init.Uniform(range=0.1))
    outgate_backward = Gate(W_in=lasagne.init.GlorotUniform(), W_hid=lasagne.init.GlorotUniform(),
                            W_cell=lasagne.init.Uniform(range=0.1))
    # according to Jozefowicz et al.(2015), init bias of forget gate to 1.
    forgetgate_backward = Gate(W_in=lasagne.init.GlorotUniform(), W_hid=lasagne.init.GlorotUniform(),
                               W_cell=lasagne.init.Uniform(range=0.1), b=lasagne.init.Constant(1.))
    # now use tanh for nonlinear function of cell, need to try pure linear cell
    cell_backward = Gate(W_in=lasagne.init.GlorotUniform(), W_hid=lasagne.init.GlorotUniform(), W_cell=None,
                         nonlinearity=nonlinearities.tanh)
    lstm_backward = lasagne.layers.LSTMLayer(incoming, num_units, mask_input=layer_mask,
                                             grad_clipping=grad_clipping, nonlinearity=nonlinearities.tanh,
                                             peepholes=peepholes, backwards=True, ingate=ingate_backward,
                                             outgate=outgate_backward, forgetgate=forgetgate_backward,
                                             cell=cell_backward, name='backward')
    # concatenate the outputs of forward and backward RNNs to combine them.
    concat = lasagne.layers.concat([lstm_forward, lstm_backward], axis=2, name="bi-lstm")
    # dropout for output
    if dropout:
        concat = lasagne.layers.DropoutLayer(concat, p=0.5)
    # the shape of BiRNN output (concat) is (batch_size, input_length, 2 * num_hidden_units)
    model = CRFLayer(concat, num_labels, mask_input=layer_mask)
    energies = lasagne.layers.get_output(model, deterministic=True)
    prediction = utils.crf_prediction(energies)
    prediction_fn = theano.function([input_var, mask_var, char_input_var], [prediction])
    return model, prediction_fn
