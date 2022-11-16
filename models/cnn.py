import numpy as np
from utils import *


class CNN:

    def __init__(self, pad, stride, pool_f=2, pool_stride=1):
        self.pad = pad
        self.stride = stride
        self.pool_f = pool_f
        self.pool_stride = pool_stride

    def zero_pad(self, X):
        """
        Add zeros around the border of images
        :param X: an array of shape (m, n_H, n_W, n_C) representing a batch of images
        :p: an integer representing the amount of padding added around the image
        :return: padded array of images
        """
        pad = self.pad
        X_pad = np.pad(X, ((0, 0), (pad, pad), (pad, pad), (0, 0)))

        return X_pad

    def conv_step(self, a_prev_slice, W, b):
        """
        Apply convolution operation defined by parameters W to a single slice of an image
        :param a_prev_slice: slice of an input data of shape (f, f, n_C_prev)
        :param W: an array of weight parameters of shape (f, f, n_C_prev)
        :param b: an array with bias parameter of shape (1, 1, 1)
        :return Z: a scalar value, the result of convolving the sliding window (W, b) on a slice of an input data
        """
        s = np.multiply(a_prev_slice, W)
        s = np.sum(s)
        Z = s + float(b)  # convert so that the result is a scalar

    def conv_forward(self, A_prev, W, b):
        """
        Forward propagation using convolution function
        :param A_prev: input data of shape (m, n_H_prev, n_W_prev, n_C_prev)
        :param W: an array of weight parameters of shape (f, f, n_C_prev, n_C)
        :param b: an array with bias parameter of shape (1, 1, 1, n_C)
        :return Z: output of convolution, shape (m, n_H, n_W, n_C)
            cache: cache of values needed for backprop
        """
        pad = self.pad
        stride = self.stride

        # Retrieve dims from A_prev's shape
        (m, n_H_prev, n_W_prev, n_C_prev) = A_prev.shape

        # Retrieve dims from filter's shape
        (f, f, n_C_prev, n_C) = W.shape

        # Define the dimensions of the output volume
        n_H = int((n_H_prev - f + 2 * pad) / stride + 1)
        n_W = int((n_W_prev - f + 2 * pad) / stride + 1)

        # Initialize the output volume with zeros
        Z = np.zeros((m, n_H, n_W, n_C))

        # Add padding to A_prev
        A_prev_pad = self.zero_pad(A_prev, pad)

        # Apply convolution to a set of images
        for i in range(m):  # loop over examples
            a_prev_pad = A_prev_pad[i]

            for h in range(n_H):  # loop over vertical axis
                vert_start = h * stride
                vert_end = vert_start + f

                for w in range(n_W):  # loop over horizontal axis
                    horiz_start = w * stride
                    horiz_end = horiz_start + f

                    for c in range(n_C):  # loop over num_filters
                        a_prev_slice = a_prev_pad[vert_start:vert_end, horiz_start:horiz_end,
                                       :]  # define the 3D slice of a_prev_pad
                        weights = W[:, :, :, c]
                        biases = b[:, :, :, c]
                        Z[i, h, w, c] = self.conv_step(a_prev_slice, weights, biases)  # get back one output neuron

        # Save information in `cache` for the backprop
        cache = (A_prev, W, b)

        return Z, cache

    def pool_forward(self, A_prev, mode='max'):
        """
        Forward pass for the pooling layer
        :param A_prev: input data of shape (m, n_H_prev, n_W_prev, n_C_prev)
        :param mode: the pooling mode ('max' or 'avg')
        :return A: output of pooling, shape (m, n_H, n_W, n_C)
            cache: cache of values needed for backprop
        """
        f = self.pool_f
        stride = self.pool_stride

        # Retrieve dims from A_prev's shape
        (m, n_H_prev, n_W_prev, n_C_prev) = A_prev.shape

        # Define the dimensions of the output volume
        n_H = int((n_H_prev - f) / stride + 1)
        n_W = int((n_W_prev - f) / stride + 1)
        n_C = n_C_prev

        # Initialize output matrix
        A = np.zeros((m, n_H, n_W, n_C))

        # Apply pooling to a set of images
        for i in range(m):  # loop over examples
            a_prev = A_prev[i]
            for h in range(n_H):  # loop over vertical axis of the output volume
                vert_start = h * stride
                vert_end = vert_start + f

                for w in range(n_W):  # loop over horizontal axis of the output volume
                    horiz_start = w * stride
                    horiz_end = horiz_start + f

                    for c in range(n_C):  # loop over the channels of the output volume
                        a_prev_slice = a_prev[vert_start:vert_end, horiz_start:horiz_end, c]
                        if mode == 'max':
                            A[i, h, w, c] = np.max(a_prev_slice)
                        elif mode == 'avg':
                            A[i, h, w, c] = np.average(a_prev_slice)

        cache = A_prev

        return A, cache

    def conv_backward(self, dZ, cache):
        """
        Compute gradients of a convolution function
        :param dZ: gradient of the cost with respect to the output of the conv layer (Z), numpy array of shape (m, n_H, n_W, n_C)
        :param cache: cache of values needed for the conv_backward(), output of conv_forward()
        :return dA_prev: gradient of the cost with respect to the input of the conv layer (A_prev),
                            numpy array of shape (m, n_H_prev, n_W_prev, n_C_prev)
                 dW: gradient of the cost with respect to the weights of the conv layer (W)
                       numpy array of shape (f, f, n_C_prev, n_C)
                 db: gradient of the cost with respect to the biases of the conv layer (b)
                       numpy array of shape (1, 1, 1, n_C)
        """
        stride = self.stride
        pad = self.pad

        # Retrieve dims from A_prev's shape
        (A_prev, W, b, hparameters) = cache

        # Define the dimensions of the output volume
        (m, n_H_prev, n_W_prev, n_C_prev) = A_prev.shape

        # Retrieve dimensions from W's shape
        (f, f, n_C_prev, n_C) = W.shape

        # Retrieve dimensions from dZ's shape
        (m, n_H, n_W, n_C) = dZ.shape

        # Initialize gradient matrices with zeros
        dA_prev = np.zeros((m, n_H_prev, n_W_prev, n_C_prev))
        dW = np.zeros((f, f, n_C_prev, n_C))
        db = np.zeros((1, 1, 1, n_C))

        # Pad A_prev and dA_prev
        A_prev_pad = self.zero_pad(A_prev, pad)
        dA_prev_pad = self.zero_pad(dA_prev, pad)

        for i in range(m):  # loop over the examples

            # Select ith training example
            a_prev_pad = A_prev_pad[i]
            da_prev_pad = dA_prev_pad[i]

            for h in range(n_H):  # loop over vertical axis of the output volume
                for w in range(n_W):  # loop over horizontal axis of the output volume
                    for c in range(n_C):  # loop over the channels of the output volume

                        # Find the corners of the current "slice"
                        vert_start = h * stride
                        vert_end = vert_start + f
                        horiz_start = w * stride
                        horiz_end = horiz_start + f

                        # Use the corners to define the slice from a_prev_pad
                        a_slice = a_prev_pad[vert_start:vert_end, horiz_start:horiz_end, :]

                        # Update gradients for the window and the filter's parameters using the code formulas given above
                        da_prev_pad[vert_start:vert_end, horiz_start:horiz_end, :] += dZ[i, h, w, c] * W[:, :, :, c]
                        dW[:, :, :, c] += a_slice * dZ[i, h, w, c]
                        db[:, :, :, c] += dZ[i, h, w, c]

            # Set the ith training example's dA_prev to the unpadded da_prev_pad
            dA_prev[i, :, :, :] = da_prev_pad[pad:-pad, pad:-pad, :]

        return dA_prev, dW, db

    def pool_backward(self, dA, cache, mode='max'):
        """
        Compute gradients for a pooling layer
        :param dA: gradient of the cost with respect to the output of the pooling layer, numpy array of shape (m, n_H, n_W, n_C)
        :param cache: cache output from the forward pass of the pooling layer
        :param mode: the pooling mode you would like to use, defined as a string ("max" or "average")
        :return dA_prev: gradient of the cost with respect to the input of the pooling layer (A_prev),
                         numpy array of shape (m, n_H_prev, n_W_prev, n_C_prev)
        """
        # Retrieve information from cache
        A_prev = cache

        # Retrieve hyperparameters from "hparameters"
        stride = self.pool_stride
        f = self.pool_f

        # Retrieve dimensions from A_prev's shape and dA's shape
        m, n_H_prev, n_W_prev, n_C_prev = A_prev.shape
        m, n_H, n_W, n_C = dA.shape

        # Initialize dA_prev with zeros
        dA_prev = np.zeros((m, n_H_prev, n_W_prev, n_C_prev))

        for i in range(m):  # loop over the training examples

            a_prev = A_prev[i]

            for h in range(n_H):  # loop on the vertical axis
                for w in range(n_W):  # loop on the horizontal axis
                    for c in range(n_C):  # loop over the channels (depth)

                        # Find the corners of the current "slice"
                        vert_start = h * stride
                        vert_end = vert_start + f
                        horiz_start = w * stride
                        horiz_end = horiz_start + f

                        # Compute the backward propagation in both modes
                        if mode == "max":

                            # Use the corners and "c" to define the current slice from a_prev
                            a_prev_slice = a_prev[vert_start:vert_end, horiz_start:horiz_end, c]

                            # Create the mask from a_prev_slice
                            mask = create_mask_from_window(a_prev_slice)

                            # Set dA_prev to be dA_prev + (the mask multiplied by the correct entry of dA)
                            dA_prev[i, vert_start: vert_end, horiz_start: horiz_end, c] += mask * dA[i, h, w, c]

                        elif mode == "average":

                            # Get the value da from dA
                            da = dA[i, h, w, c]

                            # Define the shape of the filter as fxf
                            shape = (f, f)

                            # Distribute it to get the correct slice of dA_prev. i.e. Add the distributed value of da
                            dA_prev[i, vert_start: vert_end, horiz_start: horiz_end, c] += distribute_value(da, shape)

        return dA_prev