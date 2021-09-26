import numpy as np
import matplotlib.pyplot as plt
import sklearn.preprocessing
from cvxpy import *

##############################################################
# Regression
##############################################################
class LinearRegression():

    def __init__(self, method, intercept = True, lmbda = 0):
        """
        Description: Initialise linear regression model
        
        Input:
            method: {'gradient_descent', 'linear_algebra'} determines the method used to solve for "w"
            intercept: bool, default to True, determines whether a biased is used
            lmbda: default to 0, used for ridge regression
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
        Description: train the model with either closed form solution or gradient descent
        
        Input:
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
        Description: Predicts the class for each sample in the input
        
        Input:
            X:  (n, p) normalised matrix
            
        Output:
            pred: (n,) dimensional vector where each entry is the models predicted class for the corresponding sample from X
        """
        _X = self._add_intercept(X) if self.intercept else X
        pred = _X @ self.w
        return pred

    def score(self, X, y, metric = 'mse'):
        """
        Description: reports the score of the model on input X and y
        
        Input:
            X: the predictors in shape (n, p) where n is the total number of samples
            y: the truth labels
            metric: default to "mse", determines which of {mse or mae} to report
            
        Output:
            {mse mae}: mse or mae for the input
        """
        assert metric in ['mse', 'mae'], '"score" parameter must be "mse" or "mae".'
        return ((y - self.predict(X))**2).sum() if metric == 'mse' else abs(y - self.predict(X)).sum()

class RidgeRegression(LinearRegression):

    def __init__(self, method, lmbda, intercept = True):
        """
        Description: Initialise the ridge regression model
        
        Input:
            method: {'gradient_descent', 'linear_algebra'} determines the method used to solve for "w"
            intercept: bool, default to True, determines whether a biased is used
            lmbda: regularisation parameter, set to 0 == OLS
        """
        super().__init__(method, intercept, lmbda)
        
##############################################################
# Logistic Regression
##############################################################

class LogisticRegression():

    def __init__(self, classes, intercept = True):
        """
        Description: Initialise logistic regression model
        
        Input:
            classes:  number of classes in the dataset
            intercept: default True, add intercept to model
        """
        self.classes = classes
        self.intercept = intercept

    def _add_intercept(self, X):
        interceptor = np.ones((X.shape[0], 1))
        _X = np.append(interceptor, X, axis = 1)
        return _X

    def _softmax(self, x):
        return np.exp(x) / sum(np.exp(x))

    def _grad(self, X, y, batch_size):
        gradient = np.zeros((self.w.shape))
        for i in range(batch_size):
            yi = y[[i], :].T ; xi = X[[i], :]
            yi_hat = self._softmax( self.w @ xi.T )
            gradient += (yi_hat - yi) @ xi
        return gradient

    def fit(self, X, y, epochs = 20, lr = 0.1, batch_size = 10):
        """
        Description: Trains the logistic regression model with gradient descent
        
        Input:
            X:  (n, p) normalised matrix
            y: (n,) dimensional vector where each value should be an integer indicating the class of the sample
            epochs: default to 20, determines the number of epochs to train for
            lr: default to 0.1, determines the learning rate of gradient descent
            batch_size: default to 10, determines the size of each batch for mini batch gradient descent
        """
        ohe = sklearn.preprocessing.OneHotEncoder().fit(y.reshape(-1,1))
        _y = ohe.transform(y.reshape(-1, 1))
        _X = self._add_intercept(X) if self.intercept else X
        n, p = _X.shape
        self.w = np.zeros((self.classes, p))

        assert batch_size <= n, 'Batch size must be smaller than the number of samples'
        for epoch in range(epochs):
            order = list(np.random.permutation(n))
            for i in range(n // batch_size):  
                batch = order[i * batch_size: (i + 1) * batch_size]
                batched_x = _X[batch] 
                batched_y = _y[batch]
                batch_gradient = self._grad(batched_x, batched_y,batch_size)
                self.w -= lr * batch_gradient / batch_size

        (self.coef_, self.intercept_) =  (self.w[1:], self.w[0]) if self.intercept else (self.w[1:], None)

    def predict(self, X):
        """
        Description: Predicts the class for each sample in the input
        
        Input:
            X:  (n, p) normalised matrix
            
        Output:
            pred: (n,) dimensional vector where each entry is the models predicted class for the corresponding sample from X
        """
        _X = self._add_intercept(X) if self.intercept else X
        n, p = _X.shape
        correct = 0
        pred = np.zeros((n, 1))
        for i in range(n):
            xi = _X[[i], :]
            pred[i, 0] = np.argmax(self._softmax( self.w @ xi.T ))

        return pred.flatten().astype(int)

    def score(self, X, y, verbose = False):
        """
        Description: Reports the error rate for the input data and labels
        
        Input:
            X:  (n, p) normalised matrix
            y: (n,) dimensional vector where each value should be an integer indicating the class of the sample
            
        Output:
            error_rate: a scalar value indicating the percentage of incorrect predictions
        """
        n = X.shape[0]
        y_pred = self.predict(X)
        incorrect = ((y_pred != y)>0).sum()
        error_rate = incorrect / n
        print(((y_pred - y)>0))
        if verbose:
            print(f"Error Rate: {error_rate * 100:.2f} %",)
        return error_rate
    
    
    
##############################################################
# Trees
##############################################################

import numpy as np

class _TreeNode():
    
    def __init__(self, left, right, parent, cutoff_id, cutoff_val, prediction):
        self.left = left
        self.right = right
        self.parent = parent
        self.cutoff_id = cutoff_id
        self.cutoff_val = cutoff_val
        self.prediction = prediction

    def _sqsplit(self, xTr,yTr,weights=[]):

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
        """
        Discription: Initialise a classification and regression tree (CART) model
        
        Input:
            maxdepth: default to inf, determines the depth of the tree
        """
        self.maxdepth = maxdepth
        self.root = _TreeNode(None, None, None, None, None, None)

    def fit(self, X, y, weights = None):
        """
        Discription: Initialise a classification and regression tree (CART) model
        
        Input:
            X: (n, d) matrix of data points
            y: (n,) vector of labels
            weights: default to None, used for Adaboost
        """
        n = X.shape[0]
        w = np.ones(n) / float(n) if weights is None else weights
        self.root._generate_nodes(X, y, 1, self.maxdepth, w)

    def predict(self, X):
        """
        Discription: predct y_hat from xTe using decision tree root.
        
        Input:
            X:  n x d matrix of data points
        
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
    
    def score(self, X, y, metric = 'mse'):
        """
        Description: reports the score of the model for input X and label y
        
        Input:
            X: the predictors in shape (n, p) where n is the total number of samples
            y: the truth labels
            metric: default to "mse", determines which of {mse or mae} to report
            
        Output:
            {mse mae}: mse or mae for the input
        """
        assert metric in ['mse', 'mae'], '"score" parameter must be "mse" or "mae".'
        return ((y - self.predict(X))**2).sum() if metric == 'mse' else abs(y - self.predict(X)).sum()


class RandomForest():
    def __init__(self, maxiter = 100, maxdepth=np.inf):
        """
        Description: Initialise a random forest model
        
        Input:
            maxiter: default to 100, determines the maximum number of trees
            maxdepth: default to inf, determines the depth of each tree
        """
        self.maxiter = maxiter
        self.maxdepth = maxdepth
        self.alphas = None
        self.trees = [] # a list to store all the trees, each entry is the root of the tree

    def fit(self, X, y):
        """
        Description: Fit the random forest with input X and y
        
        Input:
            X: (n, d) matrix of data points
            y: (n,) vector of labels
            
        """
        n = X.shape[0]
        for i in range(self.maxiter):
            samples = np.random.choice(n, n, True) # sample from training set, with replacment
            sample_X = X[samples] # get a new set of training X with the sampled indices
            sample_y = y[samples] # get a new set of training y with the sampled indices
            tree = CART(self.maxdepth)
            tree.fit(sample_X, sample_y)
            self.trees.append(tree.root)

    def predict(self, X):
        """
        Description: Evaluates X using trees.
        
        Input:
            X:      n x d matrix of data points
            
        Output:
            pred: (n,) dimensional vector of predictions
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
    
    def score(self, X, y, metric = 'mse'):
        """
        Description: reports the score of the model for input X and label y
        
        Input:
            X: the predictors in shape (n, p) where n is the total number of samples
            y: the truth labels
            metric: default to "mse", determines which of {mse or mae} to report
            
        Output:
            {mse mae}: mse or mae for the input
        """
        assert metric in ['mse', 'mae'], '"score" parameter must be "mse" or "mae".'
        return ((y - self.predict(X))**2).sum() if metric == 'mse' else abs(y - self.predict(X)).sum()

class AdaboostTree(RandomForest):

    def __init__(self, maxiter = 100, maxdepth=2):
        """
        Description: Learns a boosted decision tree.
        Input:
            maxiter:  default to 100, determines the maximum number of trees
            maxdepth: default to 2, determines maximum depth of a tree
        """
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
# primal/kernelSVM
##############################################################
class _kernels():
    def __init__(self, *k_param):
        self.kernels = {"rbf": self._rbf} 
                        # "polynomial": self._polynomial,
                        # "linear": self._linear}
        self.k_param = k_param

    def _rbf(self, X, Z):
        kpar = self.k_param[0]
        a, b = np.sum(X**2, axis = -1, keepdims = True), np.sum(Z**2, axis = -1, keepdims = True).T
        norm = a + b - 2*(X@Z.T)
        K = np.exp(-kpar * norm)
        return K

    # def _polynomial(self, X, Z):
    #     kpar = self.k_param[0]
    #     K = (1 + X.dot(Z.T))**kpar
    #     return K

    # def _linear(self, X, Z):
    #     K = X.dot(Z.T)
    #     return K




class kernelSVM(_kernels):

    def __init__(self, C, ktype, *k_param):
        super().__init__(*k_param)
        self.ktype = ktype
        self.computeK = self.kernels[self.ktype]
        self.C = C

    def _dualqp(self, K, yTr, C):
        """
        function alpha = dualqp(K,yTr,C)
        constructs the SVM dual formulation and uses a built-in 
        convex solver to find the optimal solution. 
        
        Input:
            K     | the (nxn) kernel matrix
            yTr   | training labels (nx1)
            C     | the SVM regularization parameter
        
        Output:
            alpha | the calculated solution vector (nx1)
        """
        y = yTr.flatten()
        N, _ = K.shape
        alpha = Variable(N)
        outerY = np.outer(y, y)
        G = outerY * K
        objective = 1/2 * quad_form(alpha, G) - sum(alpha)
        constraint = [alpha >= 0, alpha <= C, sum(multiply(alpha, y)) == 0]
        prob = Problem(Minimize(objective), constraint)
        prob.solve()

        return np.array(alpha.value).flatten()
    

    def _recoverBias(self, K,yTr,alphas,C):
        """
        function bias=recoverBias(K,yTr,alpha,C);
        Solves for the hyperplane bias term, which is uniquely specified by the 
        support vectors with alpha values 0<alpha<C
        
        INPUT:
        K : nxn kernel matrix
        yTr : nx1 input labels
        alpha  : nx1 vector of alpha values
        C : regularization constant
        
        Output:
        bias : the scalar hyperplane bias of the kernel SVM specified by alphas
        """
        distances = np.abs(alphas - C/2)
        i = np.where(distances == distances.min())[0][0]
        k = K[:, [i]]
        b = yTr[i] - np.vdot(alphas * yTr, k)

        return b

    def fit(self, xTr,yTr):
        """
        function classifier = dualSVM(xTr,yTr,C,ktype,lmbda);
        Constructs the SVM dual formulation and uses a built-in 
        convex solver to find the optimal solution. 
        
        Input:
            xTr   | training data (nxd)
            yTr   | training labels (nx1)
        """
        self.xTr, self.yTr = xTr, yTr
        K = self.computeK(self.xTr, self.xTr)
        eps = 0.00001
        K = (K + K.T) / 2 + eps * np.eye(K.shape[0])

        self.alpha = self._dualqp(K, yTr, self.C)
        self.b = self._recoverBias(K, yTr, self.alpha, self.C)

        self.alpha[self.alpha < 0.000001] = 0

    def predict(self, xTe):
        preds = (self.yTr * self.alpha).reshape((1,-1)).dot(self.computeK(self.xTr, xTe)) + self.b
        return preds

    def score(self, X, y, verbose = False):
        """
        Description: Reports the error rate for the input data and labels
        
        Input:
            X:  (n, p) normalised matrix
            y: (n,) dimensional vector where each value should be an integer indicating the class of the sample
            
        Output:
            error_rate: a scalar value indicating the percentage of incorrect predictions
        """
        n = X.shape[0]
        y_pred = self.predict(X)
        incorrect = ((y_pred != y)>0).sum()
        error_rate = incorrect / n
        print(((y_pred - y)>0))
        if verbose:
            print(f"Error Rate: {error_rate * 100:.2f} %",)
        return error_rate
        

class primalSVM():
    def __init__(self, C=1):
        """
        Description: initialise a primal SVM primal SVM.
        Input:
            C : the SVM regularization parameter
        """
        self.C = C
    def fit(self, Xtr, ytr):
        """
        Input:
            xTr   | training data (nxd)
            yTr   | training labels (n)
        
        Output:
            fun   | usage: predictions=fun(xTe); predictions.shape = (n,)
            wout  | the weight vector calculated by the solver
            bout  | the bias term calculated by the solver
        """
        self.xTr, self.yTr = Xtr, ytr.flatten()
        N, d = Xtr.shape

        w = Variable(d)
        b = Variable(1)
        objective = self.C * sum(pos(1 - multiply(self.yTr, self.xTr @ w + b))) + norm(w, 2)**2
        constraints = [w >= 0]
        prob = Problem(Minimize(objective), constraints)
        prob.solve()

        self.coef_ = w.value
        self.intercept_ = b.value

    def predict(self, X):
        pred = X.dot(self.coef_) + self.intercept_
        return pred
    
    def score(self, X, y, verbose = False):
        """
        Description: Reports the error rate for the input data and labels
        
        Input:
            X:  (n, p) normalised matrix
            y: (n,) dimensional vector where each value should be an integer indicating the class of the sample
            
        Output:
            error_rate: a scalar value indicating the percentage of incorrect predictions
        """
        n = X.shape[0]
        y_pred = self.predict(X)
        incorrect = ((y_pred != y)>0).sum()
        error_rate = incorrect / n
        print(((y_pred - y)>0))
        if verbose:
            print(f"Error Rate: {error_rate * 100:.2f} %",)
        return error_rate

class kernelRidgeRegression(_kernels):
    def __init__(self):
        pass
    
##############################################################
# Neural Network
##############################################################


class FeedForward():
    def __init__(self):
        pass
    
    
    
    
##############################################################
# k nearest neighbour
##############################################################
class _knn():
    def __init__(self, Xtr, ytr, k, method = 'l2'):
        """
        Description: Initialise the k-nearest neighbour model

        Input:
            k: the number of nearest neighbours to consider when making predictions
        """
        self.method = method
        self.measure_method = {"l2": self._l2distance}
        self.Xtr = Xtr
        self.ytr = ytr
        self.k = k

        assert self.method in list(self.measure_method.keys()), "Use {L2} norm"

    def _l2distance(self, X, Z=None):
        if Z is None:
            Z=X

        n,d1=X.shape
        m,d2=Z.shape
        assert (d1==d2), "Dimensions of input vectors must match!"

        # YOUR CODE HERE
        D = np.zeros((n, m))
        for i in range(m):
            z_i = Z[i]
            D[:, i] = np.linalg.norm(X - z_i, axis = 1)

        return D

    def _findknn(self, Xte):
        D = self.measure_method[self.method](self.Xtr, Xte)
        indices = np.argpartition(D, self.k, axis = 0)[:self.k]
        dists = np.partition(D, self.k, axis = 0)[:self.k]
        return indices, dists

class knnRegression(_knn):
    def __init__(self, Xtr, ytr, k, method = 'l2'):
        """
        Description: Initialise the k-nearest neighbour model

        Input:
            k: the number of nearest neighbours to consider when making predictions
            measure: {L2, L1}default to l2, the metrics used to search for nearest neighbours
        """
        super().__init__(Xtr, ytr, k, method)

    def predict(self, Xte):
        """
        Description: Initialise the k-nearest neighbour model

        Input:
            X: (n, d) dimensional matrix of datapoints

        Output:
            preds: predicted class of the corresponding data
        """
        indiced, dists = self._findknn(Xte)

        n,d = Xte.shape
        preds = np.zeros(n)
        ind = self._findknn(Xte)[0]
        for i in range(n):
            indices_i = list(ind[:, i])
            nn_y_i = self.ytr[indices_i]
            preds[i] = nn_y_i.mean()
        return preds


    def score(self, X, y, metric = 'mse'):
        """
        Description: reports the score of the model on input X and y
        
        Input:
            X: the predictors in shape (n, p) where n is the total number of samples
            y: the truth labels
            metric: default to "mse", determines which of {mse or mae} to report
            
        Output:
            score: mse or mae for the input
        """
        assert metric in ['mse', 'mae'], '"score" parameter must be "mse" or "mae".'
        score = ((y - self.predict(X))**2).sum() if metric == 'mse' else abs(y - self.predict(X)).sum()
        return score


class knnClassifier(_knn):
    def __init__(self, Xtr, ytr, k, method = 'l2'):
        """
        Description: Initialise the k-nearest neighbour model

        Input:
            X: (n, d) dimensional matrix of datapoints
        """
        super().__init__(Xtr, ytr, k, method)

    def _mode(self, array):
        unique = list(set(array))
        mode = None
        count = np.NINF
        for i in unique:
            current_count = (array == i).sum()
            if current_count > count:
                count = current_count
                mode = i
                
        return mode

    def predict(self, Xte):
        """
        Description: Initialise the k-nearest neighbour model

        Input:
            X: (n, d) dimensional matrix of datapoints

        Output:
            preds: predicted class of the corresponding data
        """
        indiced, dists = self._findknn(Xte)

        n,d = Xte.shape
        preds = np.zeros(n)
        ind = self._findknn(Xte)[0]
        for i in range(n):
            indices_i = list(ind[:, i])
            nn_y_i = self.ytr[indices_i]
            preds[i] = self._mode(nn_y_i)
        return preds


    def score(self, X, y, verbose = False):
        """
        Description: Reports the error rate for the input data and labels
        
        Input:
            X:  (n, p) normalised matrix
            y: (n,) dimensional vector where each value should be an integer indicating the class of the sample
            
        Output:
            error_rate: a scalar value indicating the percentage of incorrect predictions
        """
        n = X.shape[0]
        y_pred = self.predict(X)
        incorrect = ((y_pred != y)>0).sum()
        error_rate = incorrect / n
        print(((y_pred - y)>0))
        if verbose:
            print(f"Error Rate: {error_rate * 100:.2f} %",)
        return error_rate


    
