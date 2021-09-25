# ML Package
Implementation of machine learning algorithms from scratch, packaged.

## Installation
```python
pip install git+https://github.com/jytan17/ML_Package
```

## Example usecase of SVM
```python
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
 
clf = kernelSVM(C=10, ktype="rbf", 1)
clf.fit(xTr, yTr)
clf.score(xTe, yTe)

```

## Example usecase of PCA

```python


```


## Example usecase of Regression Trees with Boosting

```python
import ML

```
