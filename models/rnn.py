import numpy as np
from utils import *

class RNN:

    def __init__(self):
        pass

    def cell_forward(self, xt, a_prev, parameters):
        """
        A single forward step of the RNN-cell.

        Args:
            xt (numpy array): Input data at timestep "t", numpy array of shape (n_x, m).
            a_prev (numpy array): Hidden state at timestep "t-1", numpy array of shape (n_a, m)
            parameters (dict): Containing:
                            Wax (numpy array): Weight matrix multiplying the input, shape (n_a, n_x)
                            Waa (numpy array): Weight matrix multiplying the hidden state, shape (n_a, n_a)
                            Wya (numpy array): Weight matrix relating the hidden-state to the output, shape (n_y, n_a)
                            ba (numpy array): Bias, shape (n_a, 1)
                            by (numpy array): Bias relating the hidden-state to the output, shape (n_y, 1)
        Returns:
            a_next (numpy array): next hidden state, shape (n_a, m)
            yt_pred (numpy array): prediction at timestep "t", shape (n_y, m)
            cache (tuple):values needed for the backward pass, contains (a_next, a_prev, xt, parameters)
        """

        # Retrieve parameters from "parameters"
        Wax = parameters["Wax"]
        Waa = parameters["Waa"]
        Wya = parameters["Wya"]
        ba = parameters["ba"]
        by = parameters["by"]

        # Compute next activation state
        a_next = np.tanh(Waa @ a_prev + Wax @ xt + ba)

        # Compute output of the current cell
        yt_pred = softmax(Wya @ a_next + by)

        # Store values you need for backward propagation in cache
        cache = (a_next, a_prev, xt, parameters)

        return a_next, yt_pred, cache

    def forward(self, x, a0, parameters):
        """
        Forward propagation.

        Args:
            x (numpy array): Input data for every time-step, shape (n_x, m, T_x).
            a0 (numpy array): Initial hidden state, shape (n_a, m)
            parameters (dict): Containing:
                                Waa (numpy array): Weight matrix multiplying the hidden state, shape (n_a, n_a)
                                Wax (numpy array): Weight matrix multiplying the input, shape (n_a, n_x)
                                Wya (numpy array): Weight matrix relating the hidden-state to the output, shape (n_y, n_a)
                                ba (numpy array):  Bias, shape (n_a, 1)
                                by (numpy array): Bias relating the hidden-state to the output,  shape (n_y, 1)

        Returns:
            a (numpy array): Hidden states for every time-step, shape (n_a, m, T_x)
            y_pred (numpy array): Predictions for every time-step, shape (n_y, m, T_x)
            caches (numpy array): Tuple of values needed for the backward pass, contains (list of caches, x)
        """

        # Initialize "caches" which will contain the list of all caches
        caches = []

        # Retrieve dimensions from shapes of x and n_a, n_y
        n_x, m, T_x = x.shape
        n_y, n_a = parameters["Wya"].shape

        # Initialize "a" and "y_pred" as 3D arrays of zeros
        a = np.zeros((n_a, m, T_x))
        y_pred = np.zeros((n_y, m, T_x))

        # Initialize a_next to be a0
        a_next = a0

        # Loop over all time-steps
        for t in range(T_x):
            # Get the 2D slice 'xt' from the 3D input 'x' at time step 't'
            xt = x[:, :, t]
            # Update next hidden state, compute the prediction, get the cache
            a_next, yt_pred, cache = self.cell_forward(xt, a_next, parameters)
            # Save the value of the new "next" hidden state in a
            a[:, :, t] = a_next
            # Save the value of the prediction in y
            y_pred[:, :, t] = yt_pred
            # Append "cache" to "caches"
            caches.append(cache)

        # Store values needed for backward propagation in cache
        caches = (caches, x)

        return a, y_pred, caches

    def cell_backward(self, da_next, cache):
        """
        Backward pass for a single RNN cell.

        Args:
            da_next (numpy array): Gradient of loss w.r.t next hidden state
            cache (dict): Containing the output of cell_forward()

        Returns:
            gradients (python dict): Containing:
                                  dx (numpy array): Gradients of input data, shape (n_x, m)
                                  da_prev (numpy array): Gradients of previous hidden state, shape (n_a, m)
                                  dWax (numpy array): Gradients of input-to-hidden weights, shape (n_a, n_x)
                                  dWaa (numpy array): Gradients of hidden-to-hidden weights, shape (n_a, n_a)
                                  dba (numpy array): Gradients of bias vector, of shape (n_a, 1)
        """

        # Retrieve values from cache
        (a_next, a_prev, xt, parameters) = cache

        # Retrieve values from parameters
        Wax = parameters["Wax"]
        Waa = parameters["Waa"]
        Wya = parameters["Wya"]
        ba = parameters["ba"]
        by = parameters["by"]

        # Compute the derivative of the tanh with respect to its input
        da_dz = (1 - a_next * a_next)

        # Compute the gradient of the loss with respect to tanh input
        dz = da_next * da_dz

        # Compute the gradient of the loss with respect to Wax
        dxt = Wax.T @ dz
        dWax = dz @ xt.T

        # Compute the gradient with respect to Waa
        da_prev = Waa.T @ dz
        dWaa = dz @ a_prev.T

        # Compute the gradient with respect to b
        dba = np.sum(dz, axis=-1, keepdims=True)

        # Store the gradients in a python dictionary
        gradients = {"dxt": dxt, "da_prev": da_prev, "dWax": dWax, "dWaa": dWaa, "dba": dba}

        return gradients

    def backward(self, da, caches):
        """
        Иackward pass for a RNN over an entire sequence of input data.

        Args:
            da (numpy array): Upstream gradients of all hidden states, shape (n_a, m, T_x)
            caches (tuple): Information from the forward pass (rnn_forward)

        Returns:
            gradients (python dict) Containing:
                                      dx (numpy array): Gradient w.r.t. the input data, shape (n_x, m, T_x)
                                      da0 (numpy array): Gradient w.r.t the initial hidden state, shape (n_a, m)
                                      dWax (numpy array): Gradient w.r.t the input's weight matrix, shape (n_a, n_x)
                                      dWaa (numpy array): Gradient w.r.t the hidden state's weight matrix, shape (n_a, n_a)
                                      dba (numpy array): Gradient w.r.t the bias, shape (n_a, 1)
        """

        # Retrieve values from the first cache (t=1) of caches
        (caches, x) = caches
        (a1, a0, x1, parameters) = caches[0]

        # Retrieve dimensions from da's and x1's shapes
        n_a, m, T_x = da.shape
        n_x, m = x1.shape

        # Initialize the gradients with the right sizes
        dx = np.zeros((n_x, m, T_x))
        dWax = np.zeros((n_a, n_x))
        dWaa = np.zeros((n_a, n_a))
        dba = np.zeros((n_a, 1))
        da0 = np.zeros((n_a, m))
        da_prevt = np.zeros((n_a, m))

        # Loop through all the time steps
        for t in reversed(range(T_x)):
            # Compute gradients at time step t
            gradients = self.cell_backward(da[:, :, t] + da_prevt,
                                           caches[t])  # at every step add gradient of prev activation
            # Retrieve derivatives
            dxt, da_prevt, dWaxt, dWaat, dbat = gradients["dxt"], gradients["da_prev"], gradients["dWax"], gradients[
                "dWaa"], gradients["dba"]
            # Increment global derivatives w.r.t parameters by adding their derivative at time-step t
            dx[:, :, t] = dxt
            dWax += dWaxt
            dWaa += dWaat
            dba += dbat

            # Set da0 to the gradient of a which has been backpropagated through all time-steps
        da0 = da_prevt

        # Store the gradients in a python dictionary
        gradients = {"dx": dx, "da0": da0, "dWax": dWax, "dWaa": dWaa, "dba": dba}

        return gradients
