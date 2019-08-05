from pitrain.bfgs import BFGS, LBFGS
import itertools as itt
import numpy as np
from numpy import random as rnd
from numpy import linalg as la

#%%
## Take quadratic function
## f(x) = 1/2 * xAx + bx + c
A = np.array([
        [4, 2, 3],
        [2, 4, 1],
        [3, 1, 4]], np.float)
b = np.array([0, -1, -2], np.float)

def df(x):
    return A @ x - b
    
#%%
def test_BFGS_update():
    bfgs = BFGS(np.eye(3))
    
    x0 = rnd.rand(3)
    x1 = rnd.rand(3) + 2
    
    g0 = df(x0)
    g1 = df(x1)
    
    bfgs.update(x0, g0)
    bfgs.update(x1, g1)
    
    y = g1 - g0
    s = x1 - x0
    
    assert np.allclose(bfgs.H @ y, s)


#%%
def test_BFGS():
    bfgs = BFGS(np.eye(3))
    for i in range(10):
        for x in itt.product([0, 1], [0, 1], [0, 1],):    
            x = np.array(x, np.float)
            bfgs.update(x, df(x))
            
    print(la.inv(bfgs.H))
    assert np.allclose(la.inv(bfgs.H), A, atol=0.001)

#%%
def test_LBFGS():
    lbfgs = LBFGS(maxlen=100, theta=1)
    
    def get_H(lbfgs:LBFGS):
        n = len(lbfgs.g_last)
        return np.stack([lbfgs(g) for g in np.eye(n)])

    for i in range(10):
        for x in itt.product([0, 1], [0, 1], [0, 1],):    
            x = np.array(x, np.float)
            lbfgs.update(x, df(x))
    
    print(la.inv(get_H(lbfgs)))
    assert np.allclose(la.inv(get_H(lbfgs)), A, atol=0.001)

#%%
if __name__ == "__main__":
    test_BFGS_update()
    test_BFGS()
    test_LBFGS()