import numpy as np
import matplotlib.pyplot as plt


##############################################################
# Linear regression
##############################################################
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
        
        
##############################################################
# Trees
##############################################################

class _TreeNode():
    """
    Tree Node Class forms the basis of the following Tree based methods.
    """
    
    def __init__(self, left, right, parent, cutoff_id, cutoff_val, prediction):
        self.left = left
        self.right = right
        self.parent = parent
        self.cutoff_id = cutoff_id
        self.cutoff_val = cutoff_val
        self.prediction = prediction

    def _sqsplit(self, xTr,yTr,weights=[]):
        """Finds the best feature, cut value, and loss value.
        
        Input:
            xTr:     n x d matrix of data points
            yTr:     n-dimensional vector of labels
            weights: n-dimensional weight vector for data points
        
        Output:
            feature:  index of the best cut's feature
            cut:      cut-value of the best cut
            bestloss: loss of the best cut
        """

        N,D = xTr.shape
        assert D > 0 # must have at least one dimension
        assert N > 1 # must have at least two samples
        if weights == []: # if no weights are passed on, assign uniform weights
            weights = np.ones(N)
        weights = weights/sum(weights) # Weights need to sum to one (we just normalize them)
        bestloss = np.inf
        feature = np.inf
        cut = np.inf
        
        xTr = xTr.astype(np.longdouble)
        yTr = yTr.astype(np.longdouble)
        weights = weights.astype(np.longdouble)
        for i in range(D):
            order = np.argsort(xTr[:, i])
            ordered_x = xTr.copy()[order]
            ordered_y = yTr.copy()[order]
            ordered_weights = weights[order]
            for j in range(1, N):
                left_x = ordered_x[:j]; right_x = ordered_x[j:]
                left_y = ordered_y[:j]; right_y = ordered_y[j:]
                left_w = ordered_weights[:j]; right_w = ordered_weights[j:]

                current_cut = (left_x[-1, i] + right_x[0, i])/2
                if current_cut == left_x[-1, i]:
                    continue
                else:
                    P_l = np.sum(left_w * left_y) ; P_r = np.sum(right_w * right_y)
                    W_l = np.sum(left_w) ; W_r = np.sum(right_w)
                    Q_l = np.sum(left_w * (left_y)**2) ; Q_r = np.sum(right_w * (right_y)**2)

                    loss = Q_l - (P_l**2 / W_l) + Q_r - (P_r**2 / W_r)
                    if loss < bestloss:
                        cut = current_cut
                        feature = i
                        bestloss = loss

        return feature, cut, bestloss

    def _generate_nodes(self, xTr, yTr, current_depth, maxdepth, weights):
        n = len(yTr)
        labels = list(set(yTr))

        if n == 0:
            print('a')
            self.prediction = None
        elif sum(yTr == labels[0]) == n or sum(yTr == labels[1]) == n:
            W = np.sum(weights)
            self.prediction = sum(weights * yTr) / W
        elif current_depth >= maxdepth:
            W = np.sum(weights)
            self.prediction = sum(weights * yTr) / W
        else:
            f,c,b = self._sqsplit(xTr, yTr, weights)
            left_ind = tuple([xTr[:, f] < c]); right_ind = tuple([xTr[:, f] >= c])
            left_y = yTr[left_ind]; right_y = yTr[right_ind]
            left_x = xTr[left_ind]; right_x = xTr[right_ind]
            left_w = weights[left_ind]; right_w = weights[right_ind]
            self.cutoff_id = f ; self.cutoff_val = c

            self.left = _TreeNode(None, None, self, None, None, None)
            self.left._generate_nodes(left_x, left_y, current_depth + 1, maxdepth, left_w)

            self.right = _TreeNode(None, None, self, None, None, None)
            self.right._generate_nodes(right_x, right_y, current_depth + 1, maxdepth, right_w)

    def predict(self, x):
        if self.prediction == None:
            if x[self.cutoff_id] < self.cutoff_val:
                return self.left.predict(x)
            elif x[self.cutoff_id] >= self.cutoff_val:
                return self.right.predict(x)
        else:
            return self.prediction

class CART():
    def __init__(self, maxdepth=np.inf):
        self.maxdepth = maxdepth
        self.root = _TreeNode(None, None, None, None, None, None)

    def fit(self, X, y, weights = None):
        n = X.shape[0]
        if weights is None:
            w = np.ones(n) / float(n)
        else:
            w = weights

        self.root._generate_nodes(X, y, 1, self.maxdepth, w)

    def predict(self, X):
        """predct y_hat from xTe using decision tree root.
        
        Input:
            xTe:  n x d matrix of data points
        
        Output:
            predictions: n-dimensional vector of predictions
        """
        
        assert self.root is not None, "fit() the tree before predicting!"

        n = X.shape[0]
        predictions = np.zeros(n)
        for i in range(n):
            x_i = X[i]
            predictions[i] += self.root.predict(x_i)

        return predictions


class RandomForest():
    def __init__(self, m, maxdepth=np.inf):
        self.m = m
        self.maxdepth = maxdepth
        self.alphas = None
        self.trees = [] # a list to store all the trees, each entry is the root of the tree

    def fit(self, X, y):
        n = X.shape[0]
        for i in range(self.m):
            samples = np.random.choice(n, n, True) # sample from training set, with replacment
            sample_X = X[samples] # get a new set of training X with the sampled indices
            sample_y = y[samples] # get a new set of training y with the sampled indices
            tree = CART(self.maxdepth)
            tree.fit(sample_X, sample_y)
            self.trees.append(tree.root)

    def predict(self, X):
        """Evaluates X using trees.
        
        Input:
            trees:  list of TreeNode decision trees of length m
            X:      n x d matrix of data points
            alphas: m-dimensional weight vector
            
        Output:
            pred: n-dimensional vector of predictions
        """
        m = len(self.trees)
        n,d = X.shape
        pred = np.zeros(n)
        if not isinstance(self.alphas, list):
            self.alphas = np.ones(m) / m

        for i in range(n):
            pred_x = np.zeros(m)
            x_i = X[i]
            for j in range(m):
                pred_x[j] += self.trees[j].predict(x_i)
            pred[i] = np.sum(pred_x * self.alphas)

        return pred

class AdaboostTree(RandomForest):
    """Learns a boosted decision tree.
    Input:
        x:        n x d matrix of data points
        y:        n-dimensional vector of labels
        maxiter:  maximum number of trees
        maxdepth: maximum depth of a tree
        
    Output:
        forest: list of TreeNode decision trees of length m
        alphas: m-dimensional weight vector
        
    (note, m is at most maxiter, but may be smaller,
    as dictated by the Adaboost algorithm)
    """
    def __init__(self, maxiter = 100, maxdepth=2):
        super().__init__(maxdepth)
        self.maxdepth = maxdepth
        self.maxiter = maxiter
        self.alphas = []

    def fit(self, X, y):
        assert np.allclose(np.unique(y), np.array([-1,1])), "The labels must be -1 and 1"
        n = X.shape[0]
        self.weights = np.ones(n) / n

        for i in range(self.maxiter):
            tree = CART(self.maxdepth)
            tree.fit(X, y, weights = self.weights)
            preds = np.sign(tree.predict(X))
            epsilon = np.sum(self.weights[preds != y])

            if epsilon < 0.5:
                alpha = .5 * np.log((1 - epsilon)/epsilon)

                self.alphas.append(alpha)
                self.trees.append(tree.root)

                self.weights = (self.weights * np.exp( - alpha * preds * y)) / (2 * np.sqrt((1 - epsilon) * epsilon))
            else:
                break
##############################################################
# Kernel Methods
##############################################################

class SVM():
    def __init__(self):
        pass
                
class KernelRidgeRegression():
    def __init__(self):
        pass
    
##############################################################
# Neural Network
##############################################################


class FeedForward():
    def __init__(self):
        pass
    
