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
            self.prediction = None
        elif (yTr == labels[0]).sum() == n:
            W = np.sum(weights)
            self.prediction = sum(weights * yTr) / W
        elif (yTr == labels[1]).sum() == n:
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
