import numpy as np
import math
from utils import *


class DeepNN:

    def __init__(self, layers_dims, num_epochs=1000, optimizer='adam', batch_size=None, lr=0.0075, lmbd=None,
                 keep_probs=None, gradient_check_epochs=[], beta_1=0.9, beta_2=0.999, epsilon=1e-8, decay=False,
                 decay_rate=1, time_interval=1000, random_state=None):
        self.layers_dims = layers_dims
        self.num_epochs = num_epochs
        self.optimizer = optimizer
        self.batch_size = batch_size
        self.lr = lr
        self.lmbd = lmbd
        self.keep_probs = keep_probs
        self.set_keep_probs()
        self.gradient_check_epochs = gradient_check_epochs
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.epsilon = epsilon
        self.decay = decay
        self.decay_rate = decay_rate
        self.time_interval = time_interval
        self.random_state = random_state
        self.params = {}
        self.costs = []
        self.lr_rates = []

    def initialize_parameters(self, layers_dims):
        """
        Initialize network weights with Xavier init
        :param layers_dims: an array of dimensions for the network where the
                          first entry corresponds to the input layer size and the last entry corresponds to the output layer size
        :return: dictionary containing initialized parameters
        """

        if not self.random_state == None:
            np.random.seed(self.random_state)

        L = len(layers_dims)
        parameters = {}

        for l in range(1, L):
            parameters['W' + str(l)] = np.random.randn(layers_dims[l], layers_dims[l - 1]) * np.sqrt(
                2 / layers_dims[l - 1])
            parameters['b' + str(l)] = np.zeros((layers_dims[l], 1))

        return parameters

    def initialize_velocity(self, parameters):
        """
        Initialize velocity matrix at 0 for momentum
        :param parameters: dict with W and b parameters for each layer
        :return: dict with velocity values initialized at 0
        """
        L = len(self.params) // 2
        v = {}

        for l in range(1, L + 1):
            v["dW" + str(l)] = np.zeros_like(parameters['W' + str(l)])
            v["db" + str(l)] = np.zeros_like(parameters['b' + str(l)])

        return v

    def initialize_adam(self, parameters):
        """
        Initialize v and s matrices
        :param parameters: dict with W and b parameters for each layer
        :return: two dicts with v and s values initialized at 0
        """
        L = len(self.params) // 2
        v = {}
        s = {}

        for l in range(1, L + 1):
            v["dW" + str(l)] = np.zeros_like(parameters['W' + str(l)])
            v["db" + str(l)] = np.zeros_like(parameters['b' + str(l)])

            s["dW" + str(l)] = np.zeros_like(parameters['W' + str(l)])
            s["db" + str(l)] = np.zeros_like(parameters['b' + str(l)])

        return v, s

    def set_batch_size(self, X):
        """
        If batch_size is not given set it to the number of examples in a dataset
        :param X: input data of shape (n_x, m)
        """
        if self.batch_size == None:
            self.batch_size = X.shape[1]

    def set_keep_probs(self):
        """
        Set dropout probability for each layer
        :return: a dict with probabilities
        """
        L = len(self.layers_dims)
        keep_probs = {}

        if not self.keep_probs == None:
            assert len(self.keep_probs) == len(
                self.layers_dims) - 1, f"keep_probs length should be equal to 'layers_dims-1'"
            for i in range(1, L):
                keep_probs['D' + str(i)] = self.keep_probs[i - 1]
        else:
            for i in range(1, L):
                keep_probs['D' + str(i)] = 1.0

        self.keep_probs = keep_probs

    def gradient_check(self, parameters, X, Y, epsilon=1e-7):
        """
        Do gradient checking
        :param model: an instance of a model
        :param X: data of shape (n_x, m)
        :param Y: labels of shape (1, m)
        :param epsilon: a small number to compute approximated gradient
        """
        # Compute gradients using back propagation
        Y_hat, caches = self.forward_propagation(X, parameters)
        cost = self.compute_cost(Y_hat, Y)
        grad = self.back_propagation(Y_hat, Y, caches)
        grad = dict_to_vector(grad)

        parameters = dict_to_vector(parameters)
        num_parameters = parameters.shape[0]

        J_plus = np.zeros((num_parameters, 1))
        J_minus = np.zeros((num_parameters, 1))
        gradapprox = np.zeros((num_parameters, 1))

        # Compute gradients using numerical approximation
        for i in range(num_parameters):
            theta_plus = np.copy(parameters)
            theta_plus[i] = theta_plus[i] + epsilon

            Y_hat, _ = self.forward_propagation(X, vector_to_dict(theta_plus, self.layers_dims))
            J_plus[i] = self.compute_cost(Y_hat, Y)

            theta_minus = np.copy(parameters)
            theta_minus[i] = theta_minus[i] - epsilon

            Y_hat, _ = self.forward_propagation(X, vector_to_dict(theta_minus, self.layers_dims))
            J_minus[i] = self.compute_cost(Y_hat, Y)

            gradapprox[i] = (J_plus[i] - J_minus[i]) / (2 * epsilon)

        difference = np.linalg.norm(gradapprox - grad) / (np.linalg.norm(gradapprox) + np.linalg.norm(grad))

        if difference > 1e-7:
            print("\033[93m" + "There is a mistake in the backward propagation! difference = " + str(
                difference) + "\033[0m")
        else:
            print("\033[92m" + "Backward propagation works perfectly fine! difference = " + str(difference) + "\033[0m")

    def create_minibatches(self, X, Y):
        """
        Split the data into mini-batches
        :param X: input data of size (n_x, m)
        :param Y: target labels of size (1,m)
        """
        mini_batch_size = self.batch_size
        m = X.shape[1]

        mini_batches = []

        # Shuffle data and labels synchronously
        permutation = list(np.random.permutation(m))
        X_shuffled = X[:, permutation]
        Y_shuffled = Y[:, permutation].reshape((1, m))

        # Create batches
        num_complete_batches = math.floor(m / mini_batch_size)

        for k in range(num_complete_batches):
            mini_batch_X = X_shuffled[:, k * mini_batch_size:(k + 1) * mini_batch_size]
            mini_batch_Y = Y_shuffled[:, k * mini_batch_size:(k + 1) * mini_batch_size]

            mini_batch = (mini_batch_X, mini_batch_Y)
            mini_batches.append(mini_batch)

        # For handling the end case (if last mini-batch < batch_size)
        if m % mini_batch_size != 0:
            mini_batch_X = X_shuffled[:, num_complete_batches * mini_batch_size:]
            mini_batch_Y = Y_shuffled[:, num_complete_batches * mini_batch_size:]

            mini_batch = (mini_batch_X, mini_batch_Y)
            mini_batches.append(mini_batch)

        return mini_batches

    def forward_propagation(self, X, params):
        """
        Do forward pass with relu activation L-1 times, then follow that with a sigmoid activation
        :param X: input data, numpy array of size (n_x, m)
        :return: Y_hat, caches
        """
        L = len(params) // 2
        caches = []
        A = X

        def dropout_forward(A, l):
            """
            Add dropout to layer_activation
            :param A : layer activation
            :param l: layer number
            :return: activation array with dropout added
            """
            # Retrieve keep_prob for the layer
            keep_prob = self.keep_probs['D' + str(l)]

            # Create dropout matrix and multiply
            D = np.random.rand(A.shape[0], A.shape[1]) < keep_prob
            D = D.astype(int)
            A = np.multiply(A, D)

            # Scale values up to preserve the expected value for the layer
            A /= keep_prob

            return A, D

        def layer_forward(A_prev, W, b, activation='relu'):
            """
            Forward propagation for a single layer
            :param A_prev: output of the previous layer/input data
            :param W: weight matrix
            :param b: bias vector
            :param activation: activation function to be used in this layer, either 'sigmoid' or 'relu'
            :return: cache dict for computing the backward pass
            """
            if activation == 'relu':
                Z = W @ A_prev + b
                A = relu(Z)

            elif activation == 'sigmoid':
                Z = W @ A_prev + b
                A = sigmoid(Z)

            cache = {'A_prev': A_prev,
                     'Z': Z,
                     'W': W}

            return A, cache

        # Iterate over the first L-1 layers:
        for l in range(1, L):
            A_prev = A
            W = params['W' + str(l)]
            b = params['b' + str(l)]

            # Forward pass
            A, cache = layer_forward(A_prev, W, b, activation='relu')

            # Dropout
            A, D = dropout_forward(A, l)
            cache['D'] = D

            caches.append(cache)

        # Apply sigmoid function for the last layer
        W = params['W' + str(L)]
        b = params['b' + str(L)]

        # Forward pass
        Y_hat, cache = layer_forward(A, W, b, activation='sigmoid')

        # Dropout
        Y_hat, D = dropout_forward(Y_hat, L)
        cache['D'] = D

        # Make sure output never becomes 0 or 1
        Y_hat[Y_hat == 0] = 1e-10
        Y_hat[Y_hat == 1] = 1 - 1e-10

        caches.append(cache)
        return Y_hat, caches

    def compute_cost(self, Y_hat, Y):
        """
        Compute the cross-entropy loss
        :param Y_hat: the predicted value
        :param Y: the ground truth value
        :return: cross-entropy cost
        """
        m = Y.shape[1]

        # Compute cross entropy cost
        cross_entropy_cost = -np.sum(Y * np.log(Y_hat) + (1 - Y) * np.log(1 - Y_hat))

        # Add regularization term
        if not self.lmbd == None:
            W = [j for i, j in self.params.items() if i.startswith("W")]  # retrieve weights
            L2_reg_cost = (self.lmbd / 2) * np.sum([np.sum(np.square(w)) for w in W])  # regularization term
            cost = cross_entropy_cost + L2_reg_cost
        else:
            cost = cross_entropy_cost

        return cost

    def back_propagation(self, Y_hat, Y, caches):
        """
        Backward propagation through the whole network
        :param Y_hat: probability vector, output of the forward propagation
        :param Y: ground truth vector
        :param caches: a list of dicts where each dict stores Z,W,b,A_prev values for each layer computed during the forward pass
        :return grads: a dict with computed gradients
        """
        L = len(self.params) // 2
        m = Y_hat.shape[1]
        Y = Y.reshape(Y_hat.shape)

        def sigmoid_backward(Z):
            """
            Compute dZ of the sigmoid function
            """
            s = 1 / (1 + np.exp(-Z))
            dZ = s * (1 - s)

            return dZ

        def relu_backward(Z):
            """
            Compute dZ of the relu function
            """
            dZ = np.zeros_like(Z)
            dZ[Z <= 0] = 0
            dZ[Z > 0] = 1

            return dZ

        def dropout_backward(dA, D, l):
            """
            Add dropout during the backward pass
            :param:
            """
            # Retrieve keep_prob for the layer
            keep_prob = self.keep_probs['D' + str(l)]

            # Apply dropout
            dA = dA * D
            dA /= keep_prob

            return dA

        def layer_backward(dA, cache, activation):
            """
            Do backward propagation for a single layer
            :param cache: values stored for this layer during forward pass
            :param activation: activation function to be used in this layer, either 'sigmoid' or 'relu'
            :return: dA_prev, dW, db
            """
            A_prev = cache['A_prev']
            Z = cache['Z']
            W = cache['W']

            # Compute dZ
            if activation == 'relu':
                dZ = dA * relu_backward(Z)
            elif activation == 'sigmoid':
                dZ = dA * sigmoid_backward(Z)

            # Compute dW, db and dA_prev
            if not self.lmbd == None:
                dW = 1 / m * (dZ @ A_prev.T) + (self.lmbd / m) * W
            else:
                dW = 1 / m * (dZ @ A_prev.T)

            db = 1 / m * np.sum(dZ, axis=1, keepdims=True)
            dA_prev = W.T @ dZ

            return dA_prev, dW, db

        # A dictionary to store computed gradients
        grads = {}

        # Derivative of the loss with respect to the network output
        dY_hat = -(np.divide(Y, Y_hat) - np.divide(1 - Y, 1 - Y_hat))

        # Dropout backward
        current_cache = caches[-1]
        D = current_cache['D']
        dY_hat = dropout_backward(dY_hat, D, L)

        # Compute gradients for the output layer
        dA_prev, dW, db = layer_backward(dY_hat, current_cache, activation='sigmoid')
        grads['dA_prev' + str(L - 1)] = dA_prev
        grads['dW' + str(L)] = dW
        grads['db' + str(L)] = db

        # Compute gradients for other L-1 layers:
        for l in reversed(range(L - 1)):
            # Get cache
            current_cache = caches[l]

            # Dropout backward
            D = current_cache['D']
            dA_prev = dropout_backward(dA_prev, D, l + 1)

            # Compute gradient
            dA_prev, dW, db = layer_backward(dA_prev, current_cache, activation='relu')
            grads['dA_prev' + str(l)] = dA_prev
            grads['dW' + str(l + 1)] = dW
            grads['db' + str(l + 1)] = db

        return grads

    def update_parameters_gd(self, grads):
        """
        Update network parameters using gradient descent
        :param grads: a dict containing gradients
        :return: a dict with updated parameters
        """
        L = len(self.params) // 2
        parameters = self.params
        lr = self.lr

        for l in range(1, L + 1):
            # Update parameters
            parameters["W" + str(l)] = parameters["W" + str(l)] - lr * grads["dW" + str(l)]
            parameters["b" + str(l)] = parameters["b" + str(l)] - lr * grads["db" + str(l)]

        return parameters

    def update_parameters_momentum(self, v, grads):
        """
        Update parameters using momentum algorithm
        :param v: dict containing current velocities
        :param beta: beta hyperparameter
        """
        L = len(self.params) // 2
        parameters = self.params
        beta_1 = self.beta_1
        lr = self.lr

        for l in range(1, L + 1):
            v["dW" + str(l)] = beta_1 * v["dW" + str(l)] + (1 - beta_1) * grads["dW" + str(l)]
            v["db" + str(l)] = beta_1 * v["db" + str(l)] + (1 - beta_1) * grads["db" + str(l)]

            # Update parameters
            parameters["W" + str(l)] = parameters["W" + str(l)] - lr * v["dW" + str(l)]
            parameters["b" + str(l)] = parameters["b" + str(l)] - lr * v["db" + str(l)]

        return parameters, v

    def update_parameters_adam(self, v, s, t, grads):
        """
        Update parameters using adam algorithm
        """
        L = len(self.params) // 2
        parameters = self.params
        beta_1 = self.beta_1
        beta_2 = self.beta_2
        epsilon = self.epsilon
        lr = self.lr

        v_corrected = {}
        s_corrected = {}

        for l in range(1, L + 1):
            # Compute moving average of the gradients
            v["dW" + str(l)] = beta_1 * v["dW" + str(l)] + (1 - beta_1) * grads["dW" + str(l)]
            v["db" + str(l)] = beta_1 * v["db" + str(l)] + (1 - beta_1) * grads["db" + str(l)]

            # Compute bias-corrected first moment estimate
            v_corrected["dW" + str(l)] = v["dW" + str(l)] / (1 - beta_1 ** t)
            v_corrected["db" + str(l)] = v["db" + str(l)] / (1 - beta_1 ** t)

            # Compute moving average of the squared gradients
            s["dW" + str(l)] = beta_2 * s["dW" + str(l)] + (1 - beta_2) * np.square(grads["dW" + str(l)])
            s["db" + str(l)] = beta_2 * s["db" + str(l)] + (1 - beta_2) * np.square(grads["db" + str(l)])

            # Compute bias-corrected second moment estimate
            s_corrected["dW" + str(l)] = s["dW" + str(l)] / (1 - beta_2 ** t)
            s_corrected["db" + str(l)] = s["db" + str(l)] / (1 - beta_2 ** t)

            # Update parameters
            parameters["W" + str(l)] = parameters["W" + str(l)] - lr * (
                        v_corrected["dW" + str(l)] / (np.sqrt(s_corrected["dW" + str(l)]) + epsilon))
            parameters["b" + str(l)] = parameters["b" + str(l)] - lr * (
                        v_corrected["db" + str(l)] / (np.sqrt(s_corrected["db" + str(l)]) + epsilon))

        return parameters, v, s

    def schedule_lr_decay(self, lr_0, epoch_num):
        """
        Update the learning rate
        """
        lr = 1 / (1 + self.decay_rate * np.floor(epoch_num / self.time_interval)) * lr_0

        return lr

    def fit(self, X, Y):
        """
        Train the network
        :param X: input data of size (n_x, m)
        :param Y: vector of ground truth labels of shape (1,m)
        """
        m = X.shape[1]
        lr_0 = self.lr

        # Initialize parameters
        self.params = self.initialize_parameters(self.layers_dims)

        # Initialize the optimizer
        if self.optimizer == 'gd':
            pass

        elif self.optimizer == 'momentum':
            v = self.initialize_velocity(self.params)

        elif self.optimizer == 'adam':
            v, s = self.initialize_adam(self.params)

        # Set batch size
        self.set_batch_size(X)

        # Iterate over epochs
        for i in range(0, self.num_epochs):

            # Gradient check
            if i in self.gradient_check_epochs:
                self.gradient_check(self.params, X, Y)

            # Create mini-batches
            mini_batches = self.create_minibatches(X, Y)
            cost_total = 0
            t = i + 1

            self.lr_rates.append(self.lr)

            # Iterate over mini-batches
            for mini_batch in mini_batches:

                (mini_batch_X, mini_batch_Y) = mini_batch

                # Forward pass
                Y_hat, caches = self.forward_propagation(mini_batch_X, self.params)

                # Compute cost
                cost_total += self.compute_cost(Y_hat, mini_batch_Y)

                # Backward propagation
                grads = self.back_propagation(Y_hat, mini_batch_Y, caches)

                # Update parameters
                if self.optimizer == 'gd':
                    parameters = self.update_parameters_gd(grads)

                elif self.optimizer == 'momentum':
                    parameters, v = self.update_parameters_momentum(v, grads)

                elif self.optimizer == 'adam':
                    parameters, v, s = self.update_parameters_adam(v, s, t, grads)

                self.params = parameters

            cost_avg = cost_total / m
            self.costs.append(cost_avg)

            if self.decay:
                self.lr = self.schedule_lr_decay(lr_0, i)

    def predict(self, X):
        """
        Predict labels for examples in X using trained parameters
        :param X: input data of shape (n_x, m)
        :return: a vector of class predictions for the input data X
        """
        m = X.shape[1]

        Y_hat, _ = self.forward_propagation(X)

        predictions = np.zeros((1, m))
        predictions[Y_hat <= 0.5] = 0
        predictions[Y_hat > 0.5] = 1

        return predictions


n_x = None
n_y = None
hyperparameters = {"layers_dims": [n_x, 10, 3, n_y],
                   "num_epochs": 2500,
                   "optimizer": 'adam',
                   "batch_size": None,
                   "lr":  0.001,
                   "lmbd": 0.7,
                   "keep_probs": [1, 1, 1],
                   "gradient_check_epochs": [],
                   "beta_1": 0.9,
                   "beta_2": 0.999,
                   "epsilon": 1e-8,
                   "decay": False,
                   "decay_rate": 1,
                   "time_interval": 1000,
                   "random_state": None}

model = DeepNN(**hyperparameters)