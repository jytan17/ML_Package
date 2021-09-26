import numpy as np
import matplotlib.pyplot as plt


###################################
# Clustering
###################################
  
class kMeans():
    def __init__(self, k, random_state = 123):
        self.k = k
        self.random_state = random_state
        self.cluster_centers = None


    def _update_cluster_assignments(self, X, mu):
        """
        Description: Update cluster assignments, given the data and current cluster means
        
        INPUTS:
            X : (N, D) data matrix; each row is a D-dimensional data point
            mu : (K, D) matrix of cluster centers (means)

        OUTPUT:
            z : (N,) vector of integers indicating current cluster assignment
        """
        N, D = X.shape
        K, D = mu.shape

        z = np.zeros((N, K))
        for i in range(self.k): 
            dist_x_to_mu_i = np.linalg.norm(X - mu[i], 2, axis = 1)
            z[:, i] = dist_x_to_mu_i
        z = z.argmin(axis = 1)
        for i in range(self.k):
            if (z == i).sum() == 0:
                z[np.random.choice(N, 1)] = i
        return z
        
    def _update_cluster_means(self, X, z):
        """ 
        Description: Update the cluster means, given the data and a current cluster assignment
        
        
        INPUTS:
            X : (N, D) data matrix; each row is a D-dimensional data point
            z : (N,) vector of integers indicating current cluster assignment
            K : integer; target number of clusters

        OUTPUT:
            mu : (K, D) matrix of cluster centers (means)
        """
        N, D = X.shape
        mu = np.zeros((self.k, D))

        for i in range(self.k):
            mu[i, :] = X[z == i].mean(axis = 0)
        return mu

    def fit(self, X):
        """
        Description: Using both `update_cluster_means` and `update_cluster_assignments` to
        implement the K-means algorithm.
        
        INPUTS:
            X : (N, D) data matrix; each row is a D-dimensional data point

        """
        N, D = X.shape
        rng = np.random.default_rng(self.random_state)
        random_indices = rng.choice(N, self.k)
        mu = X[random_indices] # initialise the cluster centers to one of the k data points
        convergence = False
        z = np.zeros(N) # a vector of intergers less than k, used to indicate the cluster assignment of the datapoint
        while not convergence:
            new_z = self._update_cluster_assignments(X, mu)
            new_mu = self._update_cluster_means(X, new_z)
            convergence = (new_z==z).all()
            z = new_z
            mu = new_mu

        ss = 0
        for i in range(self.k):
            ss += ((X[z==i] - mu[i])**2).sum()
            
        self.sum_square_dif = ss
        self.cluster_centers = mu

    def predict(self, X):
        assert self.cluster_centers is not None, "Fit the data first!"
        preds = self._update_cluster_assignments(X, self.cluster_centers)
        return preds
  
class GaussianMixture():
  def __init__(self):
    pass

  
  
###################################
# Dimensionality Reduction
###################################

class PCA():
    def __init__(self, m):
        """
        Description: initialise the PCA algorithm

        Input:
            m: determines the number of pca to return
        """
        self.m = m


    def fit(self, X):
        """
        Description: This function computes the first M prinicpal components of a
        dataset X. It returns the mean of the data, the projection matrix,
        and the associated singular values.
        
        INPUT:
            X: (N, D) matrix; each row is a D-dimensional data point
        """
        N, D = X.shape
        assert self.m <= D, 'the number of principle components must be less than the number of features'

        self.x_bar = X.mean(axis = 0)
        X_tilde = X - self.x_bar
        
        if X_tilde.shape[0] == X_tilde.shape[1]:
            hermitian = (X_tilde == X.T)
        else:
            hermitian = False

        u, self.s, W = np.linalg.svd(X_tilde, 
                                full_matrices = False, 
                                hermitian = hermitian)
        self.W = W[:self.m].T

    def transform(self, X):
        """ 
        Description: Apply the PCA transformation to data (reducing dimensionality).
    
        INPUTS:
            X : (N, D) matrix; each row is a D-dimensional data point

        OUTPUT:
            Z : (N, M) matrix of transformed data
        """ 

        Z = (self.W.T @ (X - self.x_bar).T)
        return Z.T

    def inverse_transform(self, Z):
        """
        Description: Apply the PCA inverse transformation, to reconstruct the data
        from the low-dimensional projection.
        
        INPUTS:
            Z : (N, M) matrix of transformed data

        OUTPUT:
            X : (N, D) matrix; each row is a D-dimensional data point
        """
        X_hat = (self.W @ Z.T).T + self.x_bar
        return X_hat
