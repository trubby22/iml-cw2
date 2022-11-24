import pickle
import numpy as np
import pandas as pd

from torch import tensor, from_numpy, nn, optim, float32, reshape, no_grad
from torch.utils.data import TensorDataset, DataLoader
from torchvision import transforms

from sklearn.preprocessing import Normalizer, LabelBinarizer, StandardScaler, LabelEncoder
from sklearn.metrics import mean_squared_error

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

def catch_all_exceptions():
    def decorate(cls):
        for attr in cls.__dict__:
            if callable(getattr(cls, attr)):
                setattr(cls, attr, catch_exception(getattr(cls, attr)))
        return cls
    return decorate

@catch_all_exceptions()
class NeuralNetwork(nn.Module):
    def __init__(self, size):
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()
        self.layer_stack = nn.Sequential(
            nn.Linear(size, 32),
            nn.ReLU(),
            nn.Linear(32, 16), 
            nn.ReLU(),
            nn.Linear(16, 1)
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.layer_stack(x)
        return logits

@catch_all_exceptions()
class Regressor():

    def __init__(self, x, encoder = LabelEncoder(), normalizer = StandardScaler(), loss_fn=nn.MSELoss(), validator=mean_squared_error, lr=0.001, nb_epoch = 1000):
        # You can add any input parameters you need
        # Remember to set them with a default value for LabTS tests
        """ 
        Initialise the model.
          
        Arguments:
            - x {pd.DataFrame} -- Raw input data of shape 
                (batch_size, input_size), used to compute the size 
                of the network.
            - nb_epoch {int} -- number of epochs to train the network.

        """

        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################

        # Replace this code with your own
        self.encoder = encoder
        self.normalizer = normalizer

        X, _ = self._preprocessor(x, training = True)
        self.input_size = X.shape[1]
        self.output_size = 1
        self.nb_epoch = nb_epoch 
        self.model = NeuralNetwork(self.input_size)
        self.loss_fn = loss_fn
        self.validator = validator
        self.optimiser = optim.Adam(self.model.parameters(), lr=lr)
        return

        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################

    def _preprocessor(self, x, y = None, training = False):
        """ 
        Preprocess input of the network.
          
        Arguments:
            - x {pd.DataFrame} -- Raw input array of shape 
                (batch_size, input_size).
            - y {pd.DataFrame} -- Raw target array of shape (batch_size, 1).
            - training {boolean} -- Boolean indicating if we are training or 
                testing the model.

        Returns:
            - {torch.tensor} or {numpy.ndarray} -- Preprocessed input array of
              size (batch_size, input_size). The input_size does not have to be the same as the input_size for x above.
            - {torch.tensor} or {numpy.ndarray} -- Preprocessed target array of
              size (batch_size, 1).
            
        """

        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################

        # Replace this code with your own
        # Return preprocessed x and y, return None for y if it was None
        
        # return x, (y if isinstance(y, pd.DataFrame) else None)
        x_filled = x.fillna(method="bfill", axis=0, downcast='infer')

        num_cols = list(x_filled.select_dtypes(include=np.number).columns)
        cat_cols = list(x_filled.select_dtypes('object').columns)

        if training:
            num_x_norm = self.normalizer.fit_transform(x_filled[num_cols])
            cat_x_enc = self.encoder.fit_transform(x_filled[cat_cols]).reshape(-1,1)
        else:
            num_x_norm = self.normalizer.transform(x_filled[num_cols])
            cat_x_enc = self.encoder.transform(x_filled[cat_cols]).reshape(-1,1)

        x_concat = np.hstack((num_x_norm, cat_x_enc))

        if isinstance(y, pd.DataFrame):
            y = from_numpy(y.values.astype(np.float32))
        
        return from_numpy(x_concat).to(float32), y

        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################

    def _train_loop(self, x, y, batch_size, debug):
        dataset = TensorDataset(x, y)
        dl = DataLoader(dataset = dataset, batch_size = batch_size, shuffle=True)

        size = len(dataset)
        batch_total_loss = 0

        for batch, (X, Y) in enumerate(dl):
            Y = reshape(Y, (-1, 1))

            y_hat = self.model(X)
            loss = self.loss_fn(y_hat, Y)
            batch_total_loss += loss.item() * len(X)

            self.optimiser.zero_grad()
            loss.backward()
            self.optimiser.step()

            if debug and batch % 100 == 0:
                current = batch * len(X)
                print(f"loss: {loss} [{current} / {size}]")

        return batch_total_loss

        
    def fit(self, x, y, batch_size = 32, debug = False):
        """
        Regressor training function

        Arguments:
            - x {pd.DataFrame} -- Raw input array of shape 
                (batch_size, input_size).
            - y {pd.DataFrame} -- Raw output array of shape (batch_size, 1).

        Returns:
            self {Regressor} -- Trained model.

        """

        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################

        X, Y = self._preprocessor(x, y = y, training = True) # Do not forget
        
        for t in range(self.nb_epoch):
            # y_hat = self.model(X)
            # loss = self.loss_fn(y_hat, Y)

            # self.optimiser.zero_grad()
            # loss.backward()
            # self.optimiser.step()

            total_loss = 0

            if debug:
                print(f"Epoch {t + 1}\n-------------------------------")
                # print(f"loss: {loss}")
            total_loss += self._train_loop(X, Y, batch_size, debug)

            if debug:
                print(f"Average loss: {total_loss / len(X)}")


        return self

        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################

            
    def predict(self, x):
        """
        Output the value corresponding to an input x.

        Arguments:
            x {pd.DataFrame} -- Raw input array of shape 
                (batch_size, input_size).

        Returns:
            {np.ndarray} -- Predicted value for the given input (batch_size, 1).

        """

        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################

        X, _ = self._preprocessor(x, training = False) # Do not forget
        return self.model(X).detach().numpy()

        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################

    def score(self, x, y, batch_size = 32, debug = False):
        """
        Function to evaluate the model accuracy on a validation dataset.

        Arguments:
            - x {pd.DataFrame} -- Raw input array of shape 
                (batch_size, input_size).
            - y {pd.DataFrame} -- Raw output array of shape (batch_size, 1).

        Returns:
            {float} -- Quantification of the efficiency of the model.

        """

        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################

        X_norm, Y_norm = self._preprocessor(x, y = y, training = False) # Do not forget
        dataset = TensorDataset(X_norm, Y_norm)
        dl = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)

        num_batches = len(dl)
        test_loss = 0
        
        with no_grad():
            for X, Y in dl:
                Y = reshape(Y, (-1, 1))
                y_hat = self.model(X)
                test_loss += self.loss_fn(y_hat, Y).item()

        test_loss /= num_batches
        if debug:
            print(f"Avg loss: {test_loss}")
        return test_loss


        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################

@catch_exception
def save_regressor(trained_model): 
    """ 
    Utility function to save the trained regressor model in part2_model.pickle.
    """
    # If you alter this, make sure it works in tandem with load_regressor
    with open('part2_model.pickle', 'wb') as target:
        pickle.dump(trained_model, target)
    print("\nSaved model in part2_model.pickle\n")

@catch_exception
def load_regressor(): 
    """ 
    Utility function to load the trained regressor model in part2_model.pickle.
    """
    # If you alter this, make sure it works in tandem with save_regressor
    with open('part2_model.pickle', 'rb') as target:
        trained_model = pickle.load(target)
    print("\nLoaded model in part2_model.pickle\n")
    return trained_model


@catch_exception
def RegressorHyperParameterSearch(): 
    # Ensure to add whatever inputs you deem necessary to this function
    """
    Performs a hyper-parameter for fine-tuning the regressor implemented 
    in the Regressor class.

    Arguments:
        Add whatever inputs you need.
        
    Returns:
        The function should return your optimised hyper-parameters. 

    """

    #######################################################################
    #                       ** START OF YOUR CODE **
    #######################################################################

    return  # Return the chosen hyper parameters

    #######################################################################
    #                       ** END OF YOUR CODE **
    #######################################################################


@catch_exception
def example_main():

    output_label = "median_house_value"

    # Use pandas to read CSV data as it contains various object types
    # Feel free to use another CSV reader tool
    # But remember that LabTS tests take Pandas DataFrame as inputs
    data = pd.read_csv("housing.csv") 

    # Splitting input and output
    x_train = data.loc[:, data.columns != output_label]
    y_train = data.loc[:, [output_label]]

    # Training
    # This example trains on the whole available dataset. 
    # You probably want to separate some held-out data 
    # to make sure the model isn't overfitting
    regressor = Regressor(x_train, lr = 1, nb_epoch = 500)
    regressor.fit(x_train, y_train, debug=True)
    save_regressor(regressor)

    # Error
    error = regressor.score(x_train, y_train)
    print("\nRegressor error: {}\n".format(error))


if __name__ == "__main__":
    example_main()
