import numpy as np


def sigmoid(x):
    """
    A numerically stable version of the logistic sigmoid function.
    """
    pos_mask = (x >= 0)
    neg_mask = (x < 0)
    z = np.zeros_like(x)
    z[pos_mask] = np.exp(-x[pos_mask])
    z[neg_mask] = np.exp(x[neg_mask])
    top = np.ones_like(x)
    top[neg_mask] = z[neg_mask]
    return top / (1 + z)


class RNN(object):
    def __init__(self, *args):
        """
        RNN Object to serialize the NN layers
        Please read this code block and understand how it works
        """
        self.params = {}
        self.grads = {}
        self.layers = []
        self.paramName2Indices = {}
        self.layer_names = {}

        # process the parameters layer by layer
        layer_cnt = 0
        for layer in args:
            for n, v in layer.params.items():
                if v is None:
                    continue
                self.params[n] = v
                self.paramName2Indices[n] = layer_cnt
            for n, v in layer.grads.items():
                self.grads[n] = v
            if layer.name in self.layer_names:
                raise ValueError("Existing name {}!".format(layer.name))
            self.layer_names[layer.name] = True
            self.layers.append(layer)
            layer_cnt += 1
        layer_cnt = 0

    def assign(self, name, val):
        # load the given values to the layer by name
        layer_cnt = self.paramName2Indices[name]
        self.layers[layer_cnt].params[name] = val

    def assign_grads(self, name, val):
        # load the given values to the layer by name
        layer_cnt = self.paramName2Indices[name]
        self.layers[layer_cnt].grads[name] = val

    def get_params(self, name):
        # return the parameters by name
        return self.params[name]

    def get_grads(self, name):
        # return the gradients by name
        return self.grads[name]

    def gather_params(self):
        """
        Collect the parameters of every submodules
        """
        for layer in self.layers:
            for n, v in layer.params.iteritems():
                self.params[n] = v

    def gather_grads(self):
        """
        Collect the gradients of every submodules
        """
        for layer in self.layers:
            for n, v in layer.grads.iteritems():
                self.grads[n] = v

    def load(self, pretrained):
        """ 
        Load a pretrained model by names 
        """
        for layer in self.layers:
            if not hasattr(layer, "params"):
                continue
            for n, v in layer.params.iteritems():
                if n in pretrained.keys():
                    layer.params[n] = pretrained[n].copy()
                    print("Loading Params: {} Shape: {}".format(n, layer.params[n].shape))


class VanillaRNN(object):
    def __init__(self, input_dim, h_dim, init_scale=0.02, name='vanilla_rnn'):
        """
        In forward pass, please use self.params for the weights and biases for this layer
        In backward pass, store the computed gradients to self.grads
        - name: the name of current layer
        - input_dim: input dimension
        - h_dim: hidden state dimension
        - meta: to store the forward pass activations for computing backpropagation 
        """
        self.name = name
        self.wx_name = name + "_wx"
        self.wh_name = name + "_wh"
        self.b_name = name + "_b"
        self.input_dim = input_dim
        self.h_dim = h_dim
        self.params = {}
        self.grads = {}
        self.params[self.wx_name] = init_scale * np.random.randn(input_dim, h_dim)
        self.params[self.wh_name] = init_scale * np.random.randn(h_dim, h_dim)
        self.params[self.b_name] = np.zeros(h_dim)
        self.grads[self.wx_name] = None
        self.grads[self.wh_name] = None
        self.grads[self.b_name] = None
        self.meta = None
        
    def step_forward(self, x, prev_h):
        """
        x: input feature (N, D)
        prev_h: hidden state from the previous timestep (N, H)

        meta: variables needed for the backward pass
        """
        next_h, meta = None, None
        assert np.prod(x.shape[1:]) == self.input_dim, "But got {} and {}".format(
            np.prod(x.shape[1:]), self.input_dim)
        ############################################################################
        # TODO: implement forward pass of a single timestep of a vanilla RNN.      #
        # Store the results in the variable output provided above as well as       #
        # values needed for the backward pass.                                     #
        ############################################################################
        wh = self.params[self.wh_name]
        wx = self.params[self.wx_name]
        b = self.params[self.b_name]
        h_raw = np.dot(x, wx) + np.dot(prev_h, wh) + b
        next_h = np.tanh(h_raw)

        meta = (wx, wh, x, prev_h, h_raw)
        #############################################################################
        #                             END OF YOUR CODE                              #
        #############################################################################
        return next_h, meta

    def step_backward(self, dnext_h, meta):
        """
        dnext_h: gradient w.r.t. next hidden state
        meta: variables needed for the backward pass

        dx: gradients of input feature (N, D)
        dprev_h: gradients of previous hiddel state (N, H)
        dWh: gradients w.r.t. feature-to-hidden weights (D, H)
        dWx: gradients w.r.t. hidden-to-hidden weights (H, H)
        db: gradients w.r.t bias (H,)
        """
        dx, dprev_h, dWx, dWh, db = None, None, None, None, None
        #############################################################################
        # TODO: Implement the backward pass of a single timestep of a vanilla RNN.  #
        # Store the computed gradients for current layer in self.grads with         #
        # corresponding name.                                                       # 
        #############################################################################
        Wx, Wh, x, prev_h, h_raw = meta

        dh_raw = (1 - np.tanh(h_raw) ** 2) * dnext_h
        dx = np.dot(dh_raw, Wx.T)
        dprev_h = np.dot(dh_raw, Wh.T)
        dWx = np.dot(x.T, dh_raw)
        dWh = np.dot(prev_h.T, dh_raw)
        db = np.sum(dh_raw, axis=0)
        #############################################################################
        #                             END OF YOUR CODE                              #
        #############################################################################
        return dx, dprev_h, dWx, dWh, db

    def forward(self, x, h0):
        """
        x: input feature for the entire timeseries (N, T, D)
        h0: initial hidden state (N, H)
        """
        h = None
        self.meta = []
        ##############################################################################
        # TODO: Implement forward pass for a vanilla RNN running on a sequence of    #
        # input data. You should use the step_forward function that you defined      #
        # above. You can use a for loop to help compute the forward pass.            #
        ##############################################################################
        N, T, D = x.shape
        H = h0.shape[1]

        forward_steps_meta = {}
        h = np.zeros([N, T, H])

        for t in range(T):
            if t == 0:
                # h[:, t, :], forward_steps_cache[t] = step_forward(x[:, t, :], h0, Wx, Wh, b)
                h[:, t, :], forward_steps_meta[t] = self.step_forward(x[:, t, :], h0)
            else:
                # h[:, t, :], forward_steps_cache[t] = rnn_step_forward(x[:, t, :], h[:, t-1, :], Wx, Wh, b)
                h[:, t, :], forward_steps_meta[t] = self.step_forward(x[:, t, :], h[:, t-1, :])

        self.meta = (h0, forward_steps_meta, N, T, D, H)

        ##############################################################################
        #                               END OF YOUR CODE                             #
        ##############################################################################
        return h

    def backward(self, dh):
        """
        dh: gradients of hidden states for the entire timeseries (N, T, H)

        dx: gradient of inputs (N, T, D)
        dh0: gradient w.r.t. initial hidden state (N, H)
        self.grads[self.wx_name]: gradient of input-to-hidden weights (D, H)
        self.grads[self.wh_name]: gradient of hidden-to-hidden weights (H, H)
        self.grads[self.b_name]: gradient of biases (H,)
        """
        dx, dh0 = None, None
        self.grads[self.wx_name] = None
        self.grads[self.wh_name] = None
        self.grads[self.b_name] = None
        ##############################################################################
        # TODO: Implement the backward pass for a vanilla RNN running an entire      #
        # sequence of data. You should use the rnn_step_backward function that you   #
        # defined above. You can use a for loop to help compute the backward pass.   #
        # HINT: Gradients of hidden states come from two sources                     #
        ##############################################################################
        h0, forward_steps_meta, N, T, D, H = self.meta
        dx = np.zeros([N, T, D])
        dWx = np.zeros([D, H])
        dWh = np.zeros([H, H])
        db = np.zeros([H])

        dprev_h = 0
        for t in reversed(range(T)):
            dh_ag = dh[:, t, :] + dprev_h
            dx_step, dprev_h, dWx_step, dWh_step, db_step = self.step_backward(dh_ag, forward_steps_meta[t])
            dWx += dWx_step
            dWh += dWh_step
            db += db_step
            dx[:, t, :] = dx_step
        
        self.grads[self.wx_name] = dWx
        self.grads[self.wh_name] = dWh
        self.grads[self.b_name] = db

        dh0 = dprev_h
        ##############################################################################
        #                               END OF YOUR CODE                             #
        ##############################################################################
        self.meta = []
        return dx, dh0


class LSTM(object):
    def __init__(self, input_dim, h_dim, init_scale=0.02, name='lstm'):
        """
        In forward pass, please use self.params for the weights and biases for this layer
        In backward pass, store the computed gradients to self.grads
        - name: the name of current layer
        - input_dim: input dimension
        - h_dim: hidden state dimension
        - meta: to store the forward pass activations for computing backpropagation 
        """
        self.name = name
        self.wx_name = name + "_wx"
        self.wh_name = name + "_wh"
        self.b_name = name + "_b"
        self.input_dim = input_dim
        self.h_dim = h_dim
        self.params = {}
        self.grads = {}
        self.params[self.wx_name] = init_scale * np.random.randn(input_dim, 4*h_dim)
        self.params[self.wh_name] = init_scale * np.random.randn(h_dim, 4*h_dim)
        self.params[self.b_name] = np.zeros(4*h_dim)
        self.grads[self.wx_name] = None
        self.grads[self.wh_name] = None
        self.grads[self.b_name] = None
        self.meta = None
        
    def step_forward(self, x, prev_h, prev_c):
        """
        x: input feature (N, D)
        prev_h: hidden state from the previous timestep (N, H)

        meta: variables needed for the backward pass
        """
        next_h, next_c, meta = None, None, None
        #############################################################################
        # TODO: Implement the forward pass for a single timestep of an LSTM.        #
        # You may want to use the numerically stable sigmoid implementation above.  #
        #############################################################################
        Wx = self.params[self.wx_name]
        Wh = self.params[self.wh_name]
        b = self.params[self.b_name]
        # Get H for slicing up activation vector A.
        H = np.shape(prev_h)[1]
        # Calculate activation vector A=x*Wx + prev_h*Wh + b.
        a_vector = np.dot(x, Wx) + np.dot(prev_h, Wh) + b
        # Slice activation vector to get the 4 parts of it: input/forget/output/block.
        a_i = a_vector[:, 0:H]
        a_f = a_vector[:, H:2*H]
        a_o = a_vector[:, 2*H:3*H]
        a_g = a_vector[:, 3*H:]
        # Activation functions applied to our 4 gates.
        input_gate = sigmoid(a_i)
        forget_gate = sigmoid(a_f)
        output_gate = sigmoid(a_o)
        block_input = np.tanh(a_g)
        # Calculate next cell state.
        next_c = (forget_gate * prev_c) + (input_gate * block_input)
        # Calculate next hidden state.
        next_h = output_gate * np.tanh(next_c)
        # Cache variables needed for backprop.
        meta = (x, Wx, Wh, b, prev_h, prev_c, input_gate, forget_gate, output_gate, block_input, next_c, next_h)
        #############################################################################
        #                               END OF YOUR CODE                            #
        #############################################################################
        return next_h, next_c, meta
        
    def step_backward(self, dnext_h, dnext_c, meta):
        """
        dnext_h: gradient w.r.t. next hidden state
        meta: variables needed for the backward pass

        dx: gradients of input feature (N, D)
        dprev_h: gradients of previous hiddel state (N, H)
        dWh: gradients w.r.t. feature-to-hidden weights (D, H)
        dWx: gradients w.r.t. hidden-to-hidden weights (H, H)
        db: gradients w.r.t bias (H,)
        """
        dx, dh, dc, dWx, dWh, db = None, None, None, None, None, None
        #############################################################################
        # TODO: Implement the backward pass for a single timestep of an LSTM.       #
        #                                                                           #
        # HINT: For sigmoid and tanh you can compute local derivatives in terms of  #
        # the output value from the nonlinearity.                                   #
        #############################################################################
        # Reference for LSTM graph
        # http://practicalcryptography.com/miscellaneous/machine-learning/graphically-determining-backpropagation-equations/
        x, Wx, Wh, b, prev_h, prev_c, input_gate, forget_gate, output_gate, block_input, next_c, next_h = meta
        # Backprop dnext_h through the multiply gate.
        dh_tanh = dnext_h * output_gate
        da_o_partial = dnext_h * np.tanh(next_c)
        # Backprop dh_tanh through the tanh function.
        dtanh = dh_tanh * (1 - np.square(np.tanh(next_c)))
        # We add dnext_c and dtanh as during forward pass we split activation (gradients add up at forks).
        dtanh_dc = (dnext_c + dtanh)
        # Backprop dtanh_dc to calculate dprev_c.
        dprev_c = dtanh_dc * forget_gate
        # Backprop dtanh_dc towards each gate.
        da_i_partial = dtanh_dc * block_input
        da_g_partial = dtanh_dc * input_gate
        da_f_partial = dtanh_dc * prev_c
        # Backprop through gate activation functions to calculate gate derivatives.
        da_i = input_gate*(1-input_gate) * da_i_partial
        da_f = forget_gate*(1-forget_gate) * da_f_partial
        da_o = output_gate*(1-output_gate) * da_o_partial
        da_g = (1-np.square(block_input)) * da_g_partial
        # Concatenate back up our 4 gate derivatives to get da_vector.
        da_vector = np.concatenate((da_i, da_f, da_o, da_g), axis=1)
        # Backprop da_vector to get remaining gradients.
        db = np.sum(da_vector, axis=0)
        dx = np.dot(da_vector, Wx.T)
        dWx = np.dot(x.T, da_vector)
        dprev_h = np.dot(da_vector, Wh.T)
        dWh = np.dot(prev_h.T, da_vector)
        #############################################################################
        #                               END OF YOUR CODE                            #
        #############################################################################

        return dx, dprev_h, dprev_c, dWx, dWh, db

    def forward(self, x, h0):
        """
        Forward pass for an LSTM over an entire sequence of data. We assume an input
        sequence composed of T vectors, each of dimension D. The LSTM uses a hidden
        size of H, and we work over a minibatch containing N sequences. After running
        the LSTM forward, we return the hidden states for all timesteps.

        Note that the initial cell state is passed as input, but the initial cell
        state is set to zero. Also note that the cell state is not returned; it is
        an internal variable to the LSTM and is not accessed from outside.

        Inputs:
        - x: Input data of shape (N, T, D)
        - h0: Initial hidden state of shape (N, H)
        - Wx: Weights for input-to-hidden connections, of shape (D, 4H)
        - Wh: Weights for hidden-to-hidden connections, of shape (H, 4H)
        - b: Biases of shape (4H,)

        Returns a tuple of:
        - h: Hidden states for all timesteps of all sequences, of shape (N, T, H)
        - cache: Values needed for the backward pass.
        """
        h = None
        self.meta = []
        #############################################################################
        # TODO: Implement the forward pass for an LSTM over an entire timeseries.   #
        # You should use the lstm_step_forward function that you just defined.      #
        #############################################################################
        N, T, D = x.shape
        N, H = h0.shape
        h = np.zeros([N, T, H])
        # Set initial h and c states.
        prev_h = h0
        prev_c = np.zeros_like(h0)
        for time_step in range(T):
            prev_h, prev_c, cache_temp = self.step_forward(x[:,time_step,:], prev_h, prev_c)
            h[:, time_step, :] = prev_h  # Store the hidden state for this time step.
            self.meta.append(cache_temp)
        #############################################################################
        #                               END OF YOUR CODE                            #
        #############################################################################
        return h

    def backward(self, dh):
        """
        Backward pass for an LSTM over an entire sequence of data.]

        Inputs:
        - dh: Upstream gradients of hidden states, of shape (N, T, H)
        - cache: Values from the forward pass

        Returns a tuple of:
        - dx: Gradient of input data of shape (N, T, D)
        - dh0: Gradient of initial hidden state of shape (N, H)
        - dWx: Gradient of input-to-hidden weight matrix of shape (D, 4H)
        - dWh: Gradient of hidden-to-hidden weight matrix of shape (H, 4H)
        - db: Gradient of biases, of shape (4H,)
        """
        dx, dh0 = None, None
        #############################################################################
        # TODO: Implement the backward pass for an LSTM over an entire timeseries.  #
        # You should use the lstm_step_backward function that you just defined.     #
        #############################################################################
        x, Wx, Wh, b, prev_h, prev_c, input_gate, forget_gate, output_gate, block_input, next_c, next_h = self.meta[0]
        Wx = self.params[self.wx_name]
        N, T, H = dh.shape
        D, _ = Wx.shape
        dx = np.zeros([N, T, D])
        dprev_h = np.zeros_like(prev_h)
        dWx = np.zeros_like(Wx)
        dWh = np.zeros_like(Wh)
        db = np.zeros_like(b)
        # Initial gradient for cell is all zero.
        dprev_c = np.zeros_like(dprev_h)
        for time_step in reversed(range(T)):
            # Add the current timestep upstream gradient to previous calculated dh.
            cur_dh = dprev_h + dh[:,time_step,:]
            dx[:,time_step,:], dprev_h, dprev_c, dWx_temp, dWh_temp, db_temp = self.step_backward(cur_dh, dprev_c, self.meta[time_step])
            # Add gradient contributions from each time step together.
            db += db_temp
            dWh += dWh_temp
            dWx += dWx_temp
        # dh0 is the last hidden state gradient calculated.
        dh0 = dprev_h
        self.grads[self.wx_name] = dWx
        self.grads[self.wh_name] = dWh
        self.grads[self.b_name] = db
        #############################################################################
        #                               END OF YOUR CODE                            #
        #############################################################################
        self.meta = []
        return dx, dh0
            
        
class word_embedding(object):
    def __init__(self, voc_dim, vec_dim, name="we"):
        """
        In forward pass, please use self.params for the weights and biases for this layer
        In backward pass, store the computed gradients to self.grads
        - name: the name of current layer
        - v_dim: words size
        - output_dim: vector dimension
        - meta: to store the forward pass activations for computing backpropagation
        """
        self.name = name
        self.w_name = name + "_w"
        self.voc_dim = voc_dim
        self.vec_dim = vec_dim
        self.params = {}
        self.grads = {}
        self.params[self.w_name] = np.random.randn(voc_dim, vec_dim)
        self.grads[self.w_name] = None
        self.meta = None
        
    def forward(self, x):
        """
        Forward pass for word embeddings. We operate on minibatches of size N where
        each sequence has length T. We assume a vocabulary of V words, assigning each
        to a vector of dimension D.

        Inputs:
        - x: Integer array of shape (N, T) giving indices of words. Each element idx
          of x muxt be in the range 0 <= idx < V.
        - W: Weight matrix of shape (V, D) giving word vectors for all words.

        Returns a tuple of:
        - out: Array of shape (N, T, D) giving word vectors for all input words.
        - meta: Values needed for the backward pass
        """
        out, self.meta = None, None
        ##############################################################################
        # TODO: Implement the forward pass for word embeddings.                      #
        #                                                                            #
        # HINT: This can be done in one line using NumPy's array indexing.           #
        ##############################################################################
        W = self.params[self.w_name]
        #print(W.shape, x.shape)
        #print(W, x)
        out = W[x, :]
        #print(out)
        self.meta = x, W
        ##############################################################################
        #                               END OF YOUR CODE                             #
        ##############################################################################
        return out
        
    def backward(self, dout):
        """
        Backward pass for word embeddings. We cannot back-propagate into the words
        since they are integers, so we only return gradient for the word embedding
        matrix.

        HINT: Look up the function np.add.at

        Inputs:
        - dout: Upstream gradients of shape (N, T, D)
        - cache: Values from the forward pass

        Returns:
        - dW: Gradient of word embedding matrix, of shape (V, D).
        """
        self.grads[self.w_name] = None
        ##############################################################################
        # TODO: Implement the backward pass for word embeddings.                     #
        # Note that Words can appear more than once in a sequence.                   #
        # HINT: Look up the function np.add.at                                       #
        ##############################################################################
        x, W = self.meta
        dW = np.zeros_like(W)
        np.add.at(dW, x, dout)
        self.grads[self.w_name] = dW
        ##############################################################################
        #                               END OF YOUR CODE                             #
        ##############################################################################
        

class temporal_fc(object):
    def __init__(self, input_dim, output_dim, init_scale=0.02, name='t_fc'):
        """
        In forward pass, please use self.params for the weights and biases for this layer
        In backward pass, store the computed gradients to self.grads
        - name: the name of current layer
        - input_dim: input dimension
        - output_dim: output dimension
        - meta: to store the forward pass activations for computing backpropagation 
        """
        self.name = name
        self.w_name = name + "_w"
        self.b_name = name + "_b"
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.params = {}
        self.grads = {}
        self.params[self.w_name] = init_scale * np.random.randn(input_dim, output_dim)
        self.params[self.b_name] = np.zeros(output_dim)
        self.grads[self.w_name] = None
        self.grads[self.b_name] = None
        self.meta = None
        
    def forward(self, x):
        """
        Forward pass for a temporal fc layer. The input is a set of D-dimensional
        vectors arranged into a minibatch of N timeseries, each of length T. We use
        an affine function to transform each of those vectors into a new vector of
        dimension M.

        Inputs:
        - x: Input data of shape (N, T, D)
        - w: Weights of shape (D, M)
        - b: Biases of shape (M,)

        Returns a tuple of:
        - out: Output data of shape (N, T, M)
        - cache: Values needed for the backward pass
        """
        N, T, D = x.shape
        M = self.params[self.b_name].shape[0]
        out = x.reshape(N * T, D).dot(self.params[self.w_name]).reshape(N, T, M) + self.params[self.b_name]
        self.meta = [x, out]
        return out

    def backward(self, dout):
        """
        Backward pass for temporal fc layer.

        Input:
        - dout: Upstream gradients of shape (N, T, M)
        - cache: Values from forward pass

        Returns a tuple of:
        - dx: Gradient of input, of shape (N, T, D)
        - dw: Gradient of weights, of shape (D, M)
        - db: Gradient of biases, of shape (M,)
        """
        x, out = self.meta
        N, T, D = x.shape
        M = self.params[self.b_name].shape[0]

        dx = dout.reshape(N * T, M).dot(self.params[self.w_name].T).reshape(N, T, D)
        self.grads[self.w_name] = dout.reshape(N * T, M).T.dot(x.reshape(N * T, D)).T
        self.grads[self.b_name] = dout.sum(axis=(0, 1))

        return dx


class temporal_softmax_loss(object):
    def __init__(self, dim_average=True):
        """
        - dim_average: if dividing by the input dimension or not
        - dLoss: intermediate variables to store the scores
        - label: Ground truth label for classification task
        """
        self.dim_average = dim_average  # if average w.r.t. the total number of features
        self.dLoss = None
        self.label = None

    def forward(self, feat, label, mask):
        """ Some comments """
        loss = None
        N, T, V = feat.shape

        feat_flat = feat.reshape(N * T, V)
        label_flat = label.reshape(N * T)
        mask_flat = mask.reshape(N * T)

        probs = np.exp(feat_flat - np.max(feat_flat, axis=1, keepdims=True))
        probs /= np.sum(probs, axis=1, keepdims=True)
        # print(mask_flat.shape, probs.shape, label_flat.shape, N*T)
        loss = -np.sum(mask_flat * np.log(probs[np.arange(N * T), label_flat]))
        if self.dim_average:
            loss /= N

        self.dLoss = probs.copy()
        self.label = label
        self.mask = mask
        
        return loss

    def backward(self):
        N, T = self.label.shape
        dLoss = self.dLoss
        if dLoss is None:
            raise ValueError("No forward function called before for this module!")
        dLoss[np.arange(dLoss.shape[0]), self.label.reshape(N * T)] -= 1.0
        if self.dim_average:
            dLoss /= N
        dLoss *= self.mask.reshape(N * T)[:, None]
        self.dLoss = dLoss
        
        return dLoss.reshape(N, T, -1)
