import numpy as np
import matplotlib.pyplot as plt

class LinearRegression():
    """
    Solve OLS with the Normal equation or gradient descent, wrt mse loss function
    """
    def __init__(self, method, intercept = True, lmbda = 0):
        """
        Parameters
        --------
        method: {'gradient_descent', 'linear_algebra'} determines the method used to solve for "w"
        intercept: bool, default to True, determines whether a biased is used
        """
        assert method in ['gradient_descent', 'linear_algebra'], '"method" parameter must be "gradient_descent" or "linear_algebra".'
        self.method = method
        self.intercept = intercept
        self.lmbda = lmbda

    def _add_intercept(self, X):
        interceptor = np.ones((X.shape[0], 1))
        _X = np.append(interceptor, X, axis = 1)
        return _X

    def fit(self, X, y, epochs = None, lr = None):
        """
        X: the predictors in shape (n, p) where n is the total number of samples and p is the number of predictors
        y: the truth values of each sample in shape (n,)
        epochs: default to None, determines the number of epochs trained for gradient descent
        lr: default to None, determines the learning rate for gradient descent
        """
        _X = self._add_intercept(X) if self.intercept else X

        if self.method == 'gradient_descent':
            assert isinstance(epochs, int) and isinstance(lr, float), "Specify learning rate and/or epochs"
            self.w = np.random.rand(_X.shape[1])
            m = X.shape[0]
            for i in range(epochs):
                lmb = self.lmbda * np.ones_like(self.w)
                lmb[0] = 0
                gradient = (1/m) * ((_X.T @ (self.predict(X)-y)) + lmb * self.w)
                self.w -= lr * gradient


        elif self.method == 'linear_algebra':
            n, p = _X.shape
            A = np.identity(p)
            A[0,0] = 0 if self.intercept else 1
            self.w = np.linalg.inv((_X.T @ _X) + (self.lmbda * A)) @ (_X.T @ y)
            
        self.coef_ = self.w[1:] if self.intercept else self.w
        self.intercept_ = self.w[0] if self.intercept else None

    def predict(self, X):
        """
        Parameters
        --------
        X: the predictors in shape (n, p) where n is the total number of samples
        
        Returns
        --------
        y_hat, the predicted y values
        """
        _X = self._add_intercept(X) if self.intercept else X
        return _X @ self.w

    def score(self, X, y, metric = 'mse'):
        """
        Parameters
        --------
        X: the predictors in shape (n, p) where n is the total number of samples
        y: the truth labels
        
        Returns
        --------
        the mse or mae
        """
        assert metric in ['mse', 'mae'], '"score" parameter must be "mse" or "mae".'
        return ((y - self.predict(X))**2).sum() if metric == 'mse' else abs(y - self.predict(X)).sum()

class RidgeRegression(LinearRegression):
    """
    solves regression with L2 regularization
    """
    def __init__(self, method, lmbda, intercept = True):
        """
        Parameters
        --------
        method: {'gradient_descent', 'linear_algebra'} determines the method used to solve for "w"
        intercept: bool, default to True, determines whether a biased is used
        lmbda: regularisation parameter, set to 0 == OLS
        """
        super().__init__(method, intercept, lmbda)
