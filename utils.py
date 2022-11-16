from operator import itemgetter

# Basic NN
def sigmoid(Z):
    """
    Implements sigmoid activation function
    :param Z: numpy array of any shape
    :return: sigmoid output
    """
    A = 1 / (1 + np.exp(-Z))

    return A


def relu(Z):
    """
    Implements ReLU activation function
    :param Z: numpy array of any shape
    :return: relu output
    """
    A = np.maximum(0, Z)
    assert (A.shape == Z.shape), f'Incorrect shape inside ReLU function!'

    return A


def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)


def dict_to_vector(d):
    """
    Reshape a dictionary into a single vector
    :param d: a dict containing model parameters or gradients
    :return: a vector of parameters/gradients
    """
    d = {k: v for k, v in d.items() if k.startswith("W") or
         k.startswith("b") or
         k.startswith("dW") or
         k.startswith("db")}

    d = dict(sorted(d.items(), key=itemgetter(0)))
    d_1 = dict(list(d.items())[:len(d) // 2])
    d_2 = dict(list(d.items())[len(d) // 2:])
    keys = [j for i in list(zip(d_1.keys(), d_2.keys())) for j in i]
    vals = [j for i in list(zip(d_1.values(), d_2.values())) for j in i]
    d = dict(zip(keys, vals))

    count = 0
    for key, value in d.items():
        new_vector = np.reshape(value, (-1, 1))

        if count == 0:
            theta = new_vector
        else:
            theta = np.concatenate((theta, new_vector), axis=0)
        count = count + 1

    return theta


def vector_to_dict(theta, layers_dims):
    """
    Unroll vector of parameters into a dictionary
    """
    L = len(layers_dims)
    parameters = {}

    for l in range(1, L):
        n_w = layers_dims[l] * layers_dims[l - 1]
        parameters['W' + str(l)] = theta[:n_w].reshape(layers_dims[l], layers_dims[l - 1])
        theta = theta[n_w:]

        n_b = layers_dims[l]
        parameters['b' + str(l)] = theta[:n_b].reshape(layers_dims[l], 1)
        theta = theta[n_b:]

    return parameters

# CNN
# Helper functions
def create_mask_from_window(x):
    """
    Create a mask of an input x, which identifies the max number in x
    :param x: a numpy array of shape (f,f)
    :return mask: a numpy array of shape (f,f) where the max entry contains True and all other enntrties contain False
    """
    mask = (x == np.max(x))

    return mask


def distribute_value(dz, shape):
    """
    Distribute input value in the matrix of dimension shape
    :param dz: input scalar
    :param shape: the shape (n_H, n_W) of the output matrix for which we want to distribute the value of dz
    :return a: array of size (n_H, n_W) for which we distributed the value of dz
    """
    # Retrieve dims from the shape
    (n_H, n_W) = shape

    # Compute the value to distribute on the matrix
    average = dz / (n_H * n_W)

    # Create a matrix where every entry is the "average" value
    a = np.ones((n_H, n_W)) * average

    return a