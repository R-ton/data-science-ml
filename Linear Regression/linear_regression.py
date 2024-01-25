from numpy import matrix, ones, cross, concatenate, zeros, insert, dot, reshape, array, r_
from numpy.linalg import inv
import pandas as pds

class LinearRegression:
    """
        Initializes the class instance with an optional parameter add_const_b. 
        If add_const_b is True, the class instance will add the constant b. 
    """
    def __init__(self, add_const_b = False) -> None:
        self.add_b = add_const_b
        self.w = None
    """
        Adds a bias to the input data and returns the biased data.

        Parameters:
            self: the object instance
            X: input data array

        Returns:
            X_biased: the biased input data array
    """
    def add_bias(self, X):
        X_biased = X
        if(X_biased.ndim == 1):
            X_biased = reshape(X, (len(X), 1))
      
        if self.add_b:
            #get the size of the features
            (N, D) = X_biased.shape
            #contruct identity vector with the height of features
            x_offset_m = ones((N, 1), dtype='int32').reshape(-1)

            X_biased = insert(X_biased, 0, x_offset_m, axis=1)
        return X_biased


    """
        Perform a prediction using the input data X and return the dot product of the biased input data and the weight vector self.w.
        Parameters:
        self: the object instance
        X: input data matrix or array of feature

    Returns:
        y: the predicted value
    """
    def predict(self, X):
        X_biased = self.add_bias(X)
        #X.w
        return dot(X_biased, self.w)
    
    """
        Train the model using the input data X and target labels y.
        Calculate the optimal value w using the ordinary least squares (OLS) method and biased X.
        Update the model's weights and return the updated weights.
        Returns:
            w: the updated weights and OLS global minimum value
    """
    def train(self, X, y):
        X_biased = self.add_bias(X)       
        # here the best optimal value is order least square or minimum of the derivated (w)
        # w is noted as w = ( X transpose . X ) ^ -1 . X transpose . y 
        X_transpose = X_biased.T
        X_line_product = dot(X_transpose, X_biased)
        w = dot(inv(X_line_product), dot(X_transpose, y))
        self.w = w
        return self.w