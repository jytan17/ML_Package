# Machine Learning Package
Implementation of machine learning algorithms from scratch. 

This package was developed for my own educational purposes and therefore, it is not recommended for actual usage.

Each algorithm has a general API akin to that of sklearn's:
```python
from models.supervised import *     # "supervised" module contains supervised learning models
from models.unsupervised import *   # "unsupervised" module contains unsupervised learning models

model = algorithm(model_params)     # initiate a model (replace "algorithm" with PCA, LinearRegression, etc)
model.fit(Xtr, ytr)                 # fit the model to the data, ytr is not required for models in the "unsupervised" module

# some regression models will have a score method to measure the model's performance on a particular test set
model.score(Xte, yte, type)         # type can be "mae" or "mse" 

# some classification models will have a score method to measure the model's accuracy on a particular test set
model.score(Xte, yte) 

# to make predictions, use
model.predict(Xte)

# the only anomaly is the PCA model which has a transform method to project the original data into a reduced dimensionality
z = PCA.transform(Xte)
# to reconstruct z into its original format, use inver_transform
x = PCA.inverse_transform(z) 

```
## Table of Contents
1. [Algorithms](#algorithms)
2. [Installation](#installation)
3. [Example usecases](#examples)
    1. [SVM](#svm)
    2. [PCA](#pca)  

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
from models.supervised import kernelSVM

# generate a spiral dataset
clf = kernelSVM(10, "rbf", 1)           # initiate a kernelSVM classifier, specify kernel type and their corresponding kernel parameter
clf.fit(xTr, yTr)                       # fit the model; models parameters can be obtained with clf.coef_ and clf.intercept_
clf.score(xTe, yTe)                     # obtain the error rate of the model on dataset xTe, yTe
```

### PCA <a name="pca"></a>

```python
from models.unsupervised import PCA

model = PCA(m = 2)              # "m" is the number of principle components to return
model.fit(Xte)                  # calculate the PC's

z = model.transform(X) # project original data into a reduced dimensional format
X_hat = model.inverset_transform(z) # reconstruct reduced data into its original format

```

