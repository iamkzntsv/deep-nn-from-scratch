import numpy as np
from utils import *


class LSTM:

    def __init__(self):
        pass

    def cell_forward(self, xt, a_prev, c_prev, parameters):
        """
        A single forward step of the LSTM-cell.

        Args:
            xt (numpy array): Input data at timestep "t", shape (n_x, m).
            a_prev (numpy array): Hidden state at timestep "t-1", shape (n_a, m)
            c_prev (numpy array): Memory state at timestep "t-1", shape (n_a, m)
            parameters (dict): Containing:
                                Wf (numpy array): weight matrix of the forget gate, shape (n_a, n_a + n_x)
                                bf (numpy array): bias of the forget gate, shape (n_a, 1)
                                Wi (numpy array): weight matrix of the update gate, shape (n_a, n_a + n_x)
                                bi (numpy array): bias of the update gate, shape (n_a, 1)
                                Wc (numpy array): weight matrix of the first "tanh", shape (n_a, n_a + n_x)
                                bc (numpy array):  bias of the first "tanh", shape (n_a, 1)
                                Wo (numpy array): weight matrix of the output gate, shape (n_a, n_a + n_x)
                                bo (numpy array):  bias of the output gate, shape (n_a, 1)
                                Wy (numpy array): weight matrix relating the hidden-state to the output, shape (n_y, n_a)
                                by (numpy array): bias relating the hidden-state to the output, shape (n_y, 1)

        Returns:
            a_next (numpy array): Next hidden state, of shape (n_a, m)
            c_next (numpy array):  Next memory state, of shape (n_a, m)
            yt_pred (numpy array): Prediction at timestep "t", numpy array of shape (n_y, m)
            cache (numpy array):  Values needed for the backward pass, contains (a_next, c_next, a_prev, c_prev, xt, parameters)
        """

        # Retrieve parameters from "parameters"
        Wf = parameters["Wf"]  # forget gate weight
        bf = parameters["bf"]
        Wi = parameters["Wi"]  # update gate weight (notice the variable name)
        bi = parameters["bi"]  # (notice the variable name)
        Wc = parameters["Wc"]  # candidate value weight
        bc = parameters["bc"]
        Wo = parameters["Wo"]  # output gate weight
        bo = parameters["bo"]
        Wy = parameters["Wy"]  # prediction weight
        by = parameters["by"]

        # Concatenate a_prev and xt
        concat = np.concatenate([a_prev, xt], axis=0)

        # Compute values for ft, it, cct, c_next, ot, a_next
        ft = sigmoid(Wf @ concat + bf)  # forget gate
        it = sigmoid(Wi @ concat + bi)  # update gate
        cct = np.tanh(Wc @ concat + bc)  # candidate values
        c_next = ft * c_prev + it * cct  # c_t
        ot = sigmoid(Wo @ concat + bo)  # output gate
        a_next = ot * (np.tanh(c_next))  # a_t

        # Compute prediction of the LSTM cell
        yt_pred = softmax(Wy @ a_next + by)

        # Store values needed for backward propagation in cache
        cache = (a_next, c_next, a_prev, c_prev, ft, it, cct, ot, xt, parameters)

        return a_next, c_next, yt_pred, cache

    def forward(self, x, a0, parameters):
        """
        Implement the forward propagation of the recurrent neural network using an LSTM-cell described in Figure (4).

        Arguments:
        x (numpy array): Input data for every time step, shape (n_x, m, T_x).
        a0 (numpy array):  Initial hidden state, of shape (n_a, m)
        parameters (numpy array):  python dictionary containing:
                            Wf (numpy array):  weight matrix of the forget gate, shape (n_a, n_a + n_x)
                            bf (numpy array):  bias of the forget gate, shape (n_a, 1)
                            Wi (numpy array):  weight matrix of the update gate, shape (n_a, n_a + n_x)
                            bi (numpy array):  bias of the update gate, shape (n_a, 1)
                            Wc (numpy array):  weight matrix of the first "tanh", shape (n_a, n_a + n_x)
                            bc (numpy array):  bias of the first "tanh", shape (n_a, 1)
                            Wo (numpy array):  weight matrix of the output gate, (n_a, n_a + n_x)
                            bo (numpy array):  bias of the output gate, shape (n_a, 1)
                            Wy (numpy array):  weight matrix relating the hidden-state to the output, shape (n_y, n_a)
                            by (numpy array):  bias relating the hidden-state to the output, shape (n_y, 1)

        Returns:
        a (numpy array):  Hidden states for every time-step, of shape (n_a, m, T_x)
        y (numpy array):  Predictions for every time-step, shape (n_y, m, T_x)
        c (numpy array):  The value of the cell state, shape (n_a, m, T_x)
        caches (tuple):  Values needed for the backward pass, contains (list of all the caches, x)
        """

        # Initialize "caches", which will track the list of all the caches
        caches = []

        # Retrieve dimensions from shapes of x and parameters['Wy']
        n_x, m, T_x = x.shape
        n_y, n_a = parameters['Wy'].shape

        # Initialize "a", "c" and "y" with zeros
        a = np.zeros((n_a, m, T_x))
        c = np.zeros((n_a, m, T_x))
        y = np.zeros((n_y, m, T_x))

        # Initialize a_next and c_next
        a_next = a0
        c_next = np.zeros_like(a_next)

        # Loop over all time steps
        for t in range(T_x):
            # Get the 2D slice 'xt' from the 3D input 'x' at time step 't'
            xt = x[:, :, t]
            # Update next hidden state, next memory state, compute the prediction, get the cache
            a_next, c_next, yt, cache = self.cell_forward(xt, a_next, c_next, parameters)
            # Save the value of the new "next" hidden state
            a[:, :, t] = a_next
            # Save the value of the next cell state
            c[:, :, t] = c_next
            # Save the value of the prediction in y
            y[:, :, t] = yt
            # Append the cache into caches
            caches.append(cache)

        # Store values needed for backward propagation in cache
        caches = (caches, x)

        return a, y, c, caches

    def cell_backward(self, da_next, dc_next, cache):
        """
        Backward pass for the LSTM cell.

        Args:
            da_next (numpy array): Gradients of next hidden state, of shape (n_a, m)
            dc_next (numpy array): Gradients of next cell state, of shape (n_a, m)
            cache (numpy array): cache storing information from the forward pass

        Returns:
            gradients (dict): Containing:
                                dxt (numpy array): Gradient of input data at time-step t, shape (n_x, m)
                                da_prev (numpy array): Gradient w.r.t. the previous hidden state, shape (n_a, m)
                                dc_prev (numpy array): Gradient w.r.t. the previous memory state, shape (n_a, m, T_x)
                                dWf (numpy array): Gradient w.r.t. the weight matrix of the forget gate, shape (n_a, n_a + n_x)
                                dWi (numpy array): Gradient w.r.t. the weight matrix of the update gate, shape (n_a, n_a + n_x)
                                dWc (numpy array): Gradient w.r.t. the weight matrix of the memory gate, shape (n_a, n_a + n_x)
                                dWo (numpy array): Gradient w.r.t. the weight matrix of the output gate, shape (n_a, n_a + n_x)
                                dbf (numpy array): Gradient w.r.t. biases of the forget gate, shape (n_a, 1)
                                dbi (numpy array): Gradient w.r.t. biases of the update gate, shape (n_a, 1)
                                dbc (numpy array): Gradient w.r.t. biases of the memory gate, shape (n_a, 1)
                                dbo (numpy array): Gradient w.r.t. biases of the output gate, shape (n_a, 1)
        """

        # Retrieve information from "cache"
        (a_next, c_next, a_prev, c_prev, ft, it, cct, ot, xt, parameters) = cache

        # Retrieve dimensions from xt and a_next
        n_x, m = xt.shape
        n_a, m = a_next.shape

        # Compute gates related derivatives
        dot = da_next * np.tanh(c_next) * ot * (1 - ot)
        dcct = (dc_next * it + ot * (1 - np.square(np.tanh(c_next))) * it * da_next) * (1 - np.square(cct))
        dit = (dc_next * cct + ot * (1 - np.square(np.tanh(c_next))) * cct * da_next) * it * (1 - it)
        dft = (dc_next * c_prev + ot * (1 - np.square(np.tanh(c_next))) * c_prev * da_next) * ft * (1 - ft)

        # Compute parameters related derivatives
        dWf = dft @ np.concatenate((a_prev, xt), axis=0).T
        dWi = dit @ np.concatenate((a_prev, xt), axis=0).T
        dWc = dcct @ np.concatenate((a_prev, xt), axis=0).T
        dWo = dot @ np.concatenate((a_prev, xt), axis=0).T
        dbf = np.sum(dft, axis=1, keepdims=True)
        dbi = np.sum(dit, axis=1, keepdims=True)
        dbc = np.sum(dcct, axis=1, keepdims=True)
        dbo = np.sum(dot, axis=1, keepdims=True)

        # Compute derivatives w.r.t previous hidden state, previous memory state and input
        da_prev = parameters['Wf'][:, :n_a].T @ dft + parameters['Wi'][:, :n_a].T @ dit + parameters['Wc'][:,
                                                                                          :n_a].T @ dcct + parameters[
                                                                                                               'Wo'][:,
                                                                                                           :n_a].T @ dot
        dc_prev = dc_next * ft + ot * (1 - np.square(np.tanh(c_next))) * ft * da_next
        dxt = parameters['Wf'][:, n_a:].T @ dft + parameters['Wi'][:, n_a:].T @ dit + parameters['Wc'][:,
                                                                                      n_a:].T @ dcct + parameters['Wo'][
                                                                                                       :, n_a:].T @ dot

        # Save gradients in dictionary
        gradients = {"dxt": dxt, "da_prev": da_prev, "dc_prev": dc_prev, "dWf": dWf, "dbf": dbf, "dWi": dWi, "dbi": dbi,
                     "dWc": dWc, "dbc": dbc, "dWo": dWo, "dbo": dbo}

        return gradients

    def backward(self, da, caches):
        """
        Backward pass for the RNN with LSTM-cell (over a whole sequence).

        Args:
            da (numpy array): Gradients w.r.t the hidden states, numpy-array of shape (n_a, m, T_x)
            caches (numpy array): cache storing information from the forward pass (lstm_forward)

        Returns:
            gradients (dict): Containing:
                                dx (numpy array): Gradient of inputs, of shape (n_x, m, T_x)
                                da0 (numpy array): Gradient w.r.t. the previous hidden state, numpy array of shape (n_a, m)
                                dWf (numpy array): Gradient w.r.t. the weight matrix of the forget gate, numpy array of shape (n_a, n_a + n_x)
                                dWi (numpy array): Gradient w.r.t. the weight matrix of the update gate, numpy array of shape (n_a, n_a + n_x)
                                dWc (numpy array): Gradient w.r.t. the weight matrix of the memory gate, numpy array of shape (n_a, n_a + n_x)
                                dWo (numpy array): Gradient w.r.t. the weight matrix of the output gate, numpy array of shape (n_a, n_a + n_x)
                                dbf (numpy array): Gradient w.r.t. biases of the forget gate, of shape (n_a, 1)
                                dbi (numpy array): Gradient w.r.t. biases of the update gate, of shape (n_a, 1)
                                dbc (numpy array): Gradient w.r.t. biases of the memory gate, of shape (n_a, 1)
                                dbo (numpy array): Gradient w.r.t. biases of the output gate, of shape (n_a, 1)
        """

        # Retrieve values from the first cache of caches.
        (caches, x) = caches
        (a1, c1, a0, c0, f1, i1, cc1, o1, x1, parameters) = caches[0]

        # Retrieve dimensions from da's and x1's shapes
        n_a, m, T_x = da.shape
        n_x, m = x1.shape

        # Initialize the gradients with the right sizes
        dx = np.zeros((n_x, m, T_x))
        da0 = np.zeros((n_a, m))
        da_prevt = np.zeros((n_a, m))
        dc_prevt = np.zeros((n_a, m))
        dWf = np.zeros((n_a, n_a + n_x))
        dWi = np.zeros((n_a, n_a + n_x))
        dWc = np.zeros((n_a, n_a + n_x))
        dWo = np.zeros((n_a, n_a + n_x))
        dbf = np.zeros((n_a, 1))
        dbi = np.zeros((n_a, 1))
        dbc = np.zeros((n_a, 1))
        dbo = np.zeros((n_a, 1))

        # Loop back over the whole sequence
        for t in reversed(range(T_x)):
            # Compute all gradients using cell_backward
            gradients = self.lstm_cell_backward(da[:, :, t] + da_prevt, dc_prevt, caches[t])
            # Store or add the gradient to the parameters previous step's gradient
            da_prevt = gradients['da_prev']
            dc_prevt = gradients['dc_prev']
            dx[:, :, t] = gradients['dxt']
            dWf += gradients['dWf']
            dWi += gradients['dWi']
            dWc += gradients['dWc']
            dWo += gradients['dWo']
            dbf += gradients['dbf']
            dbi += gradients['dbi']
            dbc += gradients['dbc']
            dbo += gradients['dbo']
        # Set the first activation's gradient to the backpropagated gradient da_prev.
        da0 = da_prevt

        # Store the gradients in a python dictionary
        gradients = {"dx": dx, "da0": da0, "dWf": dWf, "dbf": dbf, "dWi": dWi, "dbi": dbi,
                     "dWc": dWc, "dbc": dbc, "dWo": dWo, "dbo": dbo}

        return gradients