import numpy as np
import pickle
import math
import functools

import traceback


def catch_exception(f):
    @functools.wraps(f)
    def func(*args, **kwargs):
        try:
            return f(*args, **kwargs)
        except Exception as e:
            print("Exception caught: {}".format(e))
            traceback.print_exc()
    return func


def xavier_init(size, gain=1.0):
    """
    Xavier initialization of network weights.

    Arguments:
        - size {tuple} -- size of the network to initialise.
        - gain {float} -- gain for the Xavier initialisation.

    Returns:
        {np.ndarray} -- values of the weights.
    """
    low = -gain * np.sqrt(6.0 / np.sum(size))
    high = gain * np.sqrt(6.0 / np.sum(size))
    return np.random.uniform(low=low, high=high, size=size)


class Layer:
    """
    Abstract layer class.
    """

    def __init__(self, *args, **kwargs):
        raise NotImplementedError()

    @catch_exception
    def forward(self, *args, **kwargs):
        raise NotImplementedError()

    @catch_exception
    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    @catch_exception
    def backward(self, *args, **kwargs):
        raise NotImplementedError()

    @catch_exception
    def update_params(self, *args, **kwargs):
        pass


class MSELossLayer(Layer):
    """
    MSELossLayer: Computes mean-squared error between y_pred and y_target.
    """
    @catch_exception
    def __init__(self):
        self._cache_current = None

    @staticmethod   
    @catch_exception
    def _mse(y_pred, y_target):
        return np.mean((y_pred - y_target) ** 2)

    @staticmethod   
    @catch_exception
    def _mse_grad(y_pred, y_target):
        return 2 * (y_pred - y_target) / len(y_pred)
    @catch_exception
    def forward(self, y_pred, y_target):
        self._cache_current = y_pred, y_target
        return self._mse(y_pred, y_target)
    @catch_exception
    def backward(self):
        return self._mse_grad(*self._cache_current)


class CrossEntropyLossLayer(Layer):
    """
    CrossEntropyLossLayer: Computes the softmax followed by the negative 
    log-likelihood loss.
    """
    @catch_exception
    def __init__(self):
        self._cache_current = None

    @staticmethod   
    @catch_exception
    def softmax(x):
        numer = np.exp(x - x.max(axis=1, keepdims=True))
        denom = numer.sum(axis=1, keepdims=True)
        return numer / denom
    @catch_exception
    def forward(self, inputs, y_target):
        assert len(inputs) == len(y_target)
        n_obs = len(y_target)
        probs = self.softmax(inputs)
        self._cache_current = y_target, probs

        out = -1 / n_obs * np.sum(y_target * np.log(probs))
        return out
    @catch_exception
    def backward(self):
        y_target, probs = self._cache_current
        n_obs = len(y_target)
        return -1 / n_obs * (y_target - probs)


class SigmoidLayer(Layer):
    """
    SigmoidLayer: Applies sigmoid function elementwise.
    """
    @catch_exception
    def __init__(self):
        """ 
        Constructor of the Sigmoid layer.
        """
        self._cache_current = None
    @catch_exception
    def forward(self, x):
        """ 
        Performs forward pass through the Sigmoid layer.

        Logs information needed to compute gradient at a later stage in
        `_cache_current`.

        Arguments:
            x {np.ndarray} -- Input array of shape (batch_size, n_in).

        Returns:
            {np.ndarray} -- Output array of shape (batch_size, n_out)
        """
        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################
        self._cache_current = x
        return self.sigmoid(x)

        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################
    @catch_exception
    def backward(self, grad_z):
        """
        Given `grad_z`, the gradient of some scalar (e.g. loss) with respect to
        the output of this layer, performs back pass through the layer (i.e.
        computes gradients of loss with respect to parameters of layer and
        inputs of layer).

        Arguments:
            grad_z {np.ndarray} -- Gradient array of shape (batch_size, n_out).

        Returns:
            {np.ndarray} -- Array containing gradient with respect to layer
                input, of shape (batch_size, n_in).
        """
        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################
        in_arr = self._cache_current
        return grad_z * self.sigmoid_derivative(in_arr)

        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################
    @catch_exception
    def sigmoid_derivative(self, x):
        return self.sigmoid(x) * (1 - self.sigmoid(x))

    @staticmethod   
    @catch_exception
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))


class ReluLayer(Layer):
    """
    ReluLayer: Applies Relu function elementwise.
    """
    @catch_exception
    def __init__(self):
        """
        Constructor of the Relu layer.
        """
        self._cache_current = None
    @catch_exception
    def forward(self, x):
        """ 
        Performs forward pass through the Relu layer.

        Logs information needed to compute gradient at a later stage in
        `_cache_current`.

        Arguments:
            x {np.ndarray} -- Input array of shape (batch_size, n_in).

        Returns:
            {np.ndarray} -- Output array of shape (batch_size, n_out)
        """
        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################
        x: np.ndarray
        (batch_size_in, n_in) = x.shape
        self._cache_current = x
        res = self.softmax(x)
        (batch_size_out, n_out) = res.shape
        assert batch_size_in == batch_size_out
        return res

        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################
    @catch_exception
    def backward(self, grad_z):
        """
        Given `grad_z`, the gradient of some scalar (e.g. loss) with respect to
        the output of this layer, performs back pass through the layer (i.e.
        computes gradients of loss with respect to parameters of layer and
        inputs of layer).

        Arguments:
            grad_z {np.ndarray} -- Gradient array of shape (batch_size, n_out).

        Returns:
            {np.ndarray} -- Array containing gradient with respect to layer
                input, of shape (batch_size, n_in).
        """
        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################
        in_arr = self._cache_current
        return grad_z * self.softmax_derivative(in_arr)

        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################
    @catch_exception
    def softmax_derivative(self, x):
        return np.vectorize(self.softmax_derivative_helper)(x)
    @catch_exception
    def softmax(self, x):
        return np.vectorize(self.softmax_helper)(x)

    @staticmethod   
    @catch_exception
    def softmax_helper(x):
        return x if x > 0 else 0

    @staticmethod   
    @catch_exception
    def softmax_derivative_helper(x):
        return 1 if x > 0 else 0


class LinearLayer(Layer):
    """
    LinearLayer: Performs affine transformation of input.
    """
    @catch_exception
    def __init__(self, n_in, n_out):
        """
        Constructor of the linear layer.

        Arguments:
            - n_in {int} -- Number (or dimension) of inputs.
            - n_out {int} -- Number (or dimension) of outputs.
        """
        self.n_in = n_in
        self.n_out = n_out

        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################
        self._W = np.zeros(shape=(n_in, n_out))
        self._b = np.zeros((1, n_out))

        self._cache_current = None
        self._grad_W_current = None
        self._grad_b_current = None

        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################
    @catch_exception
    def forward(self, x):
        """
        Performs forward pass through the layer (i.e. returns Wx + b).

        Logs information needed to compute gradient at a later stage in
        `_cache_current`.

        Arguments:
            x {np.ndarray} -- Input array of shape (batch_size, n_in).

        Returns:
            {np.ndarray} -- Output array of shape (batch_size, n_out)
        """
        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################
        x: np.ndarray
        (batch_size_in, n_in) = x.shape
        assert n_in == self.n_in

        batch_b = np.repeat(self._b, batch_size_in, axis=0)
        z = x @ self._W + batch_b

        (batch_size_out, n_out) = z.shape
        assert batch_size_in == batch_size_out
        assert n_out == self.n_out

        self._cache_current = x, z
        return z
        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################
    @catch_exception
    def backward(self, grad_z):
        """
        Given `grad_z`, the gradient of some scalar (e.g. loss) with respect to
        the output of this layer, performs back pass through the layer (i.e.
        computes gradients of loss with respect to parameters of layer and
        inputs of layer).

        Arguments:
            grad_z {np.ndarray} -- Gradient array of shape (batch_size, n_out).

        Returns:
            {np.ndarray} -- Array containing gradient with respect to layer
                input, of shape (batch_size, n_in).
        """
        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################
        grad_z: np.ndarray
        (batch_size_in, n_out) = grad_z.shape
        assert n_out == self.n_out

        x, z = self._cache_current
        self._grad_W_current = x.T @ grad_z
        self._grad_b_current = np.ones(batch_size_in).T @ grad_z
        res = grad_z @ self._W.T

        (batch_size_out, n_in) = res.shape
        assert batch_size_in == batch_size_out
        assert n_in == self.n_in

        return res

        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################
    @catch_exception
    def update_params(self, learning_rate):
        """
        Performs one step of gradient descent with given learning rate on the
        layer's parameters using currently stored gradients.

        Arguments:
            learning_rate {float} -- Learning rate of update step.
        """
        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################
        self._W = self._W - learning_rate * self._grad_W_current

        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################


class MultiLayerNetwork(object):
    """
    MultiLayerNetwork: A network consisting of stacked linear layers and
    activation functions.
    """
    @catch_exception
    def __init__(self, input_dim, neurons, activations):
        """
        Constructor of the multi layer network.

        Arguments:
            - input_dim {int} -- Number of features in the input (excluding 
                the batch dimension).
            - neurons {list} -- Number of neurons in each linear layer 
                represented as a list. The length of the list determines the 
                number of linear layers.
            - activations {list} -- List of the activation functions to apply 
                to the output of each linear layer.
        """
        self.input_dim = input_dim
        self.neurons = neurons
        self.activations = activations

        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################
        str_to_layer = {
            'identity': LinearLayer,
            'relu': ReluLayer,
            'sigmoid': SigmoidLayer,
        }
        neurons_temp: list[int] = [x for x in neurons]
        neurons_temp.insert(0, input_dim)
        linear_dims = list(zip(neurons_temp, neurons))
        self._layers = []
        for i in range(len(linear_dims)):
            self._layers.append(LinearLayer(*linear_dims[i]))
            activation_cls = str_to_layer[self.activations[i]]
            if activation_cls != LinearLayer:
                self._layers.append(activation_cls())
        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################
    @catch_exception
    def forward(self, x):
        """
        Performs forward pass through the network.

        Arguments:
            x {np.ndarray} -- Input array of shape (batch_size, input_dim).

        Returns:
            {np.ndarray} -- Output array of shape (batch_size,
                #_neurons_in_final_layer)
        """
        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################
        x: np.ndarray
        res = x
        layer: Layer
        for layer in self._layers:
            res: np.ndarray = layer.forward(res)
        return res

        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################
    @catch_exception
    def __call__(self, x):
        return self.forward(x)
    @catch_exception
    def backward(self, grad_z):
        """
        Performs backward pass through the network.

        Arguments:
            grad_z {np.ndarray} -- Gradient array of shape (batch_size,
                #_neurons_in_final_layer).

        Returns:
            {np.ndarray} -- Array containing gradient with respect to layer
                input, of shape (batch_size, input_dim).
        """
        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################
        res = grad_z
        layer: Layer
        reverse_layers = [x for x in self._layers]
        reverse_layers.reverse()
        for layer in reverse_layers:
            res = layer.backward(res)
        return res

        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################
    @catch_exception
    def update_params(self, learning_rate):
        """
        Performs one step of gradient descent with given learning rate on the
        parameters of all layers using currently stored gradients.

        Arguments:
            learning_rate {float} -- Learning rate of update step.
        """
        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################
        layer: Layer
        for layer in self._layers:
            layer.update_params(learning_rate)

        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################


def save_network(network, fpath):
    """
    Utility function to pickle `network` at file path `fpath`.
    """
    with open(fpath, "wb") as f:
        pickle.dump(network, f)


def load_network(fpath):
    """
    Utility function to load network found at file path `fpath`.
    """
    with open(fpath, "rb") as f:
        network = pickle.load(f)
    return network


class Trainer(object):
    """
    Trainer: Object that manages the training of a neural network.
    """
    @catch_exception
    def __init__(
            self,
            network,
            batch_size,
            nb_epoch,
            learning_rate,
            loss_fun,
            shuffle_flag,
    ):
        """
        Constructor of the Trainer.

        Arguments:
            - network {MultiLayerNetwork} -- MultiLayerNetwork to be trained.
            - batch_size {int} -- Training batch size.
            - nb_epoch {int} -- Number of training epochs.
            - learning_rate {float} -- SGD learning rate to be used in training.
            - loss_fun {str} -- Loss function to be used. Possible values: mse,
                cross_entropy.
            - shuffle_flag {bool} -- If True, training data is shuffled before
                training.
        """
        self.network = network
        self.batch_size = batch_size
        self.nb_epoch = nb_epoch
        self.learning_rate = learning_rate
        self.loss_fun = loss_fun
        self.shuffle_flag = shuffle_flag

        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################
        loss_fun_to_layer = {
            'mse': MSELossLayer,
            'cross_entropy': CrossEntropyLossLayer,
        }
        self._loss_layer: Layer = loss_fun_to_layer[loss_fun]()
        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################

    @staticmethod   
    @catch_exception
    def shuffle(input_dataset, target_dataset):
        """
        Returns shuffled versions of the inputs.

        Arguments:
            - input_dataset {np.ndarray} -- Array of input features, of shape
                (#_data_points, n_features) or (#_data_points,).
            - target_dataset {np.ndarray} -- Array of corresponding targets, of
                shape (#_data_points, #output_neurons).

        Returns: 
            - {np.ndarray} -- shuffled inputs.
            - {np.ndarray} -- shuffled_targets.
        """
        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################
        indices = np.arange(input_dataset.shape[0])
        np.random.shuffle(indices)
        return input_dataset[indices], target_dataset[indices]

        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################
    @catch_exception
    def train(self, input_dataset, target_dataset):
        """
        Main training loop. Performs the following steps `nb_epoch` times:
            - Shuffles the input data (if `shuffle` is True)
            - Splits the dataset into batches of size `batch_size`.
            - For each batch:
                - Performs forward pass through the network given the current
                batch of inputs.
                - Computes loss.
                - Performs backward pass to compute gradients of loss with
                respect to parameters of network.
                - Performs one step of gradient descent on the network
                parameters.

        Arguments:
            - input_dataset {np.ndarray} -- Array of input features, of shape
                (#_training_data_points, n_features).
            - target_dataset {np.ndarray} -- Array of corresponding targets, of
                shape (#_training_data_points, #output_neurons).
        """
        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################
        input_dataset: np.ndarray
        target_dataset: np.ndarray
        self.network: MultiLayerNetwork
        self._loss_layer: Layer

        for i in range(self.nb_epoch):
            if self.shuffle_flag:
                input_dataset, target_dataset = self.shuffle(input_dataset, target_dataset)
            no_splits = int(input_dataset.shape[0] / self.batch_size)
            input_batches = np.split(input_dataset, no_splits)
            target_batches = np.split(target_dataset, no_splits)
            for input_batch, expected_output in zip(input_batches, target_batches):
                actual_output = self.network.forward(input_batch)
                self._loss_layer.forward(actual_output, expected_output)
                grad_z = self._loss_layer.backward()
                # grad_z = np.gradient(actual_output, loss, axis=0)
                self.network.backward(grad_z)
                self.network.update_params(self.learning_rate)

        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################
    @catch_exception
    def eval_loss(self, input_dataset, target_dataset):
        """
        Function that evaluate the loss function for given data. Returns
        scalar value.

        Arguments:
            - input_dataset {np.ndarray} -- Array of input features, of shape
                (#_evaluation_data_points, n_features).
            - target_dataset {np.ndarray} -- Array of corresponding targets, of
                shape (#_evaluation_data_points, #output_neurons).
        
        Returns:
            a scalar value -- the loss
        """
        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################
        self.network: MultiLayerNetwork
        actual_output = self.network.forward(input_dataset)
        return self._loss_layer.forward(actual_output, target_dataset)

        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################


class Preprocessor(object):
    """
    Preprocessor: Object used to apply "preprocessing" operation to datasets.
    The object can also be used to revert the changes.
    """
    @catch_exception
    def __init__(self, data):
        """
        Initializes the Preprocessor according to the provided dataset.
        (Does not modify the dataset.)

        Arguments:
            data {np.ndarray} dataset used to determine the parameters for
            the normalization.
        """
        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################
        data: np.ndarray
        self.a = 0
        self.b = 1
        self.x_min = np.min(data)
        self.x_max = np.max(data)

        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################
    @catch_exception
    def apply(self, data):
        """
        Apply the pre-processing operations to the provided dataset.

        Arguments:
            data {np.ndarray} dataset to be normalized.

        Returns:
            {np.ndarray} normalized dataset.
        """
        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################
        return self.a + (data - self.x_min) * (self.b - self.a) / (self.x_max - self.x_min)

        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################
    @catch_exception
    def revert(self, data):
        """
        Revert the pre-processing operations to retrieve the original dataset.

        Arguments:
            data {np.ndarray} dataset for which to revert normalization.

        Returns:
            {np.ndarray} reverted dataset.
        """
        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################
        dirty_res = self.x_min + (data - self.a) * (self.x_max - self.x_min) / (self.b - self.a)

        def round_num(x):
            return round(x, 2)

        return np.vectorize(round_num)(dirty_res)

        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################


def example_main():
    input_dim = 4
    neurons = [16, 3]
    activations = ["relu", "identity"]
    net = MultiLayerNetwork(input_dim, neurons, activations)

    dat = np.loadtxt("iris.dat")
    np.random.shuffle(dat)

    x = dat[:, :4]
    y = dat[:, 4:]

    split_idx = int(0.8 * len(x))

    x_train = x[:split_idx]
    y_train = y[:split_idx]
    x_val = x[split_idx:]
    y_val = y[split_idx:]

    prep_input = Preprocessor(x_train)

    x_train_pre = prep_input.apply(x_train)
    x_val_pre = prep_input.apply(x_val)

    # Test

    assert (prep_input.revert(x_train_pre) == x_train).all()
    assert (prep_input.revert(x_val_pre) == x_val).all()

    # End test

    trainer = Trainer(
        network=net,
        batch_size=8,
        nb_epoch=1000,
        learning_rate=0.01,
        loss_fun="cross_entropy",
        shuffle_flag=True,
    )

    trainer.train(x_train_pre, y_train)
    print("Train loss = ", trainer.eval_loss(x_train_pre, y_train))
    print("Validation loss = ", trainer.eval_loss(x_val_pre, y_val))

    preds = net(x_val_pre).argmax(axis=1).squeeze()
    targets = y_val.argmax(axis=1).squeeze()
    accuracy = (preds == targets).mean()
    print("Validation accuracy: {}".format(accuracy))


if __name__ == "__main__":
    example_main()
