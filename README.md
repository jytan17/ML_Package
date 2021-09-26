# Machine Learning Package
Implementation of machine learning algorithms from scratch. 

This package was developed for my own educational purposes and therefore, it is not recommended for actual usage.
## Table of Contents
1. [Algorithms](#algorithms)
2. [Installation](#installation)
3. [Example usecases](#examples)
    1. [SVM](#svm)
    2. [PCA](#pca)
    3. [Adaboosted Trees](#tree)   

## Algorithms (Implemented/planned) <a name="algorithms"></a>
- [x] Linear Regression
- [x] Ridge Regression
- [ ] Lasso Regression
- [ ] Kernelised Regression
- [x] Logistic Regression
- [x] Classification and Regression Tree (CART)
- [x] Random Forest
- [x] Adaboosted Trees
- [x] Support Vector Machine (SVM)
- [x] Kernelised SVM
- [x] K-Nearest Neighbours
- [x] Feedforward Network
- [ ] Recurrent Neural Network (Vanilla)
- [x] PCA
- [x] K-means
- [ ] Gaussian Mixture Model

Some interesting experiments to test my implementations were documented here "https://jytan17.github.io/Machine-Learning-package/"!

## Installation <a name="installation"></a>
Using pip:
```python
pip install git+https://github.com/jytan17/ML_Package
```

## Example usecases <a name="examples"></a>


### kernelised SVM <a name="svm"></a>
```python
import numpy as np
from models.supervised import kernelSVM

# generate a spiral dataset
def spiraldata(N=300):
    r = np.linspace(1,2*np.pi,N)
    xTr1 = np.array([np.sin(2.*r)*r, np.cos(2*r)*r]).T
    xTr2 = np.array([np.sin(2.*r+np.pi)*r, np.cos(2*r+np.pi)*r]).T
    xTr = np.concatenate([xTr1, xTr2], axis=0)
    yTr = np.concatenate([np.ones(N), -1 * np.ones(N)])
    xTr = xTr + np.random.randn(xTr.shape[0], xTr.shape[1])*0.2
    
    xTe = xTr[::2,:]
    yTe = yTr[::2]
    xTr = xTr[1::2,:]
    yTr = yTr[1::2]
    
    return xTr,yTr,xTe,yTe
    
 xTr,yTr,xTe,yTe=spiraldata()


clf = kernelSVM(10, "rbf", 1)   # initiate a kernelSVM classifier, specify kernel type and their corresponding kernel parameter
clf.fit(xTr, yTr)                       # fit the model; models parameters can be obtained with clf.coef_ and clf.intercept_
clf.score(xTe, yTe)                     # obtain the error rate of the model on dataset xTe, yTe
```

### PCA <a name="pca"></a>

```python


```


### Regression Trees with adaboost <a name="tree"></a>

```python
import ML

```
