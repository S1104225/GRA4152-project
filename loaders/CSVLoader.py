## 
#  This module defines the CSVLoader class.
#
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

## A superclass that defines main methods and properties of all CSV loaders
#
class CSVLoader:
    ## Constructs a CSVLoader object and loads the dataset
    #  @param x_names the names of the exogenous variables
    #  @param y_names the names of the endogenous variables
    #
    def __init__(self, x_names, y_names):
        # Creates instance variables
        self._df = pd.DataFrame([])
        self._x = np.array([])
        self._y = np.array([])
        self._x_names = np.array(x_names)
        self._y_names = np.array(y_names)

        # Loads the data inside a specific subclass
        self._load()

    ## Returns the matrix of exogenous variables
    #  @return matrix of Xs
    #
    @property
    def x(self):
        return self._x
    
    ## Sets a new value for the matrix of exogenous variables
    #  @param new_x new value for the matrix of Xs
    #
    @x.setter
    def x(self, new_x):
        assert isinstance(new_x, np.ndarray), 'Expected numpy array as a new value for the matrix of Xs'
        self._x = new_x
    
    ## Returns the matrix of endogenous varibles
    #  @return maxtrix of Ys
    #
    @property
    def y(self):
        return self._y
    
    ## Sets a new value for the matrix of endogenous variables
    #  @param new_x new value for the matrix of Ys
    #
    @y.setter
    def y(self, new_y):
        assert isinstance(new_y, np.ndarray), 'Expected numpy array as a new value for the matrix of Ys'
        self._y = new_y
    
    ## Returns the transposed matrix of exogenous varibles
    #  @return transposed matrix of Xs
    #
    @property
    def x_transpose(self):
        return self._x.T
    
    ## Returns the transposed matrix of endogenous varibles
    #  @return transposed matrix of Ys
    #
    @property
    def y_transpose(self):
        return self._y.T
    
    ## Adds a vector of 1s to the matrix X
    #
    def add_constant(self):
        assert (len(self.x)) > 0, 'Data has not been loaded yet'

        # Creates a nested list of 1s with the same dimension as data
        const_vector = np.array([ np.ones(self.x.shape[1]) ])
        self.x = np.concatenate((const_vector, self.x))


    ## The abstract method to _load csv files 
    #
    def _load(self):
        # Checks that variables with names from x_names and y_name are actually inside the dataset
        assert np.all(np.isin(self._x_names, self._df.keys())), 'Wrong x_names provided'
        assert np.all(np.isin(self._y_names, self._df.keys())), 'Wrong y_names provided'

        # Saves transposed data since in future we require data in the following format: [p, N]
        self.x = self._df.loc[:, self._x_names].to_numpy().T
        self.y = self._df.loc[:, self._y_names].to_numpy().T
        print(f'Successfully loaded dataset: X -> {self.x.shape} | y -> {self.y.shape}')

    ## Splits the dataset into testing and training parts
    #  @param test_size a fraction of data that will be used for testing. If it isn't a float in (0, 1)
    #  the whole dataset will be used for training and testing
    #  @return tuple with the following data: x_train, x_test, y_train, y_test
    #
    def test_train_split(self, test_size, random_state):
        if (isinstance(test_size, float) and 0 < test_size < 1):
            # We use map function to transpose matrices since train_test_split can only split data in the following format: [N, p]
            return map(lambda x: x.T, train_test_split(self.x_transpose, self.y_transpose, test_size=test_size, random_state=random_state))
        else:
            return self.x, self.x, self.y, self.y