from collections import deque
import numpy as np
from numpy import ndarray, dot
from typing import Deque

class BFGS:
    """
    For updating the inverse Hessian `H`, we use the update formula (3) from:
    Nocedal, J. (1980). "Updating Quasi-Newton Matrices with Limited Storage". Mathematics of Computation. 35 (151): 773–782. 
    https://www.ams.org/journals/mcom/1980-35-151/S0025-5718-1980-0572855-7/home.html
    
    We just rearrange it a bit in order to compute efficiently:
    $$H_{new} = H - \rho (A + A^T) + \rho^2 (s^T H s) y y^T + \rho s s^T$$
    where $A = y (s^T H)$ and $\rho = 1 / (s^T y)$.
    """
    
    
    x_last: ndarray # last observed point $x$
    g_last: ndarray # last observed gradient $g$
    H: ndarray # estimated INVERSE Hessian
    
    def __init__(self, H0: ndarray):
        self.x_last = None
        self.g_last = None
        self.H = H0
        
    def update(self, x, g) -> None:
        if self.x_last is not None:
            s = x - self.x_last
            y = g - self.g_last
            sy = dot(s, y)
            assert sy > 0
            rho = 1 / sy
            
            v = np.eye(len(s)) - rho * y[:, None] * s[None, :]
            
            self.H = v.T @ self.H @ v + rho * s[:, None] * s[None, :]
            
            
#            A = y[:, None] * (s @ self.H)[None, :]
#            self.H += (
#                - rho * (A + A.T) 
#                + rho**2 * (s @ self.H @ s) * y[:, None] * y[None, :]
#                + rho * s[:, None] * s[None, :]
#            )
        
        self.x_last = x
        self.g_last = g
    
    def __call__(self, g):
        return self.H @ g

class LBFGS:
    """
    We follow notations of the original paper:
    Nocedal, J. (1980). "Updating Quasi-Newton Matrices with Limited Storage". Mathematics of Computation. 35 (151): 773–782. 
    https://www.ams.org/journals/mcom/1980-35-151/S0025-5718-1980-0572855-7/home.html
    
    We take `H0 := theta * identity`
    """
    theta: float
    s: Deque[ndarray] # stored vectors $x_k - x_{k-1}$
    y: Deque[ndarray] # stored vectors $g_k - g_{k-1}$
    rho: Deque[float] # stored scalars  `1 / dot(y[k], s[k])`
    x_last: np.ndarray # last observed point $x$
    g_last: np.ndarray # last observed gradient $g$
    
    
    def __init__(self, maxlen:int, theta:float):
        self.theta = theta
        self.s = deque([], maxlen)
        self.y = deque([], maxlen)
        self.rho = deque([], maxlen)
        self.x_last = None
        self.g_last = None
        
    def update(self, x, g):
        if self.x_last is not None:
            self.s.append(x - self.x_last)
            self.y.append(g - self.g_last)
            
            sy = dot(self.s[-1], self.y[-1])
            assert sy > 0
            self.rho.append(1 / sy)
        
        self.x_last = x
        self.g_last = g
        
    def __call__(self, g):
        
        s = self.s
        y = self.y
        rho = self.rho
        
        n = len(s)
        alpha = [None] * n
        q = np.array(g)
        
        for i in reversed(range(n)):
            alpha[i] = rho[i] * dot(s[i], q)
            q = q - alpha[i] * y[i]
            
        r = self.theta * q
        for i in range(n):
            beta = rho[i] * dot(y[i], r) 
            r = r + (alpha[i] - beta) * s[i]
           
        return r

