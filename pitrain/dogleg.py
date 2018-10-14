import numpy as np
import numpy.linalg as la

from types import SimpleNamespace

import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

from .misc import BFGS_update_B

######################
## Auxilliary methods for quadratic functions

def norm(v, M):
    return (v @ M @ v )**(1/2)


def quadratic_restrict_to_line(a, u, B, g = None, c = 0):
    """Restrict `Q: p -> c + g^T p + 1/2 * p^T B p` to line `t -> a + tu`.
    
    Args:
        a: point in $R^n$
        u: vector in $R^n$
        c: scalar
        g: covector in $R^n$
        B: matrix in $R^{n\times n}$
        
    Returns:
        alpha, beta, gamma: scalars s.t. `t -> alpha t^2 + beta t + gamma` is the restriction.
    """
    if g is None:
        g = np.zeros([B.shape[0]])
    
    ##
    alpha = 1/2 * u @ B @ u
    beta = g @ u + a @ B @ u
    gamma = c + g @ a + 1/2 * a @ B @ a 
    
    return alpha, beta, gamma

def solve_quadratic_1d(a, b, c):
    """Solve $a t^2 + b t + c = 0."""
    
    D = b**2 - 4* a * c
    assert D >= 0, "The quadratic function does not have real solutions."
    
    return (-b + D**(1/2)) / (2 * a), (-b - D**(1/2)) / (2 * a) 
    

#solve_quadratic_1d(1, -1, 0)

def minim_quadr_on_segment(segment, B, g, c ):
    """Minimize quadratic function `Q: p -> c + g^T p + 1/2 * p^T B p` on line `t -> a + t(b - a)`.
    
    Args:
        segment: list of points `a, b` in $R^n$
        c: scalar
        g: covector in $R^n$
        B: matrix in $R^{n\times n}$
        
    Returns:
        x_min: point in $R^n$
        Q(x_min): scalar
        t_min: scalar; `x_min = a + t_min * (b - a)`
    """
    
    a, b = segment

    u = b - a 
    ## The restriction to line `t -> a + tu` is `R: t -> alpha t^2 + beta t + gamma`.
    alpha, beta, gamma = quadratic_restrict_to_line(a = a, u = u, B = B, g = g, c = c)
    
    def R(t):
        return alpha * t**2 + beta *t + gamma

    ##
    t_stationary = - beta / (2 * alpha)

    ## force `t_stationary` into interval [0, 1]
    t_stationary = max(0, min(t_stationary, 1))
    
    ## calculate the values at `[0, t_stationary, 1]` and choose the minimal
    ttt = np.array([0, t_stationary, 1])
    i = R(ttt).argmin()
    
#     def m(p):
#         return c + g @ p + 1/2 * p @ B @ p

#     print("{} t^2 + {} t + {}".format(alpha, beta, gamma), ttt, R(ttt))
#     print([m(a + t * u ) for t in ttt])
    
    return a + ttt[i] * u, R(ttt[i]), ttt[i]
        
#minim_quadr_on_segment(
#    segment = [np.array([0]), np.array([2])],
#    c = 0, g = np.array([-1]), B = np.eye(1)
#)

def quadratic_stationary(B, g):
    """Fin stationary point of quadratic function `Q: p -> c + g^T p + 1/2 * p^T B p` in $R^n$.
    
    Args:
        c: scalar
        g: covector in $R^n$
        B: matrix in $R^{n\times n}$
        
    Returns:
        x: point in $R^n$
    """   
    return - la.pinv(B) @ g

#quadratic_stationary(
#    g = np.array([-1]), B = np.eye(1)
#)


######################################################
## Dogleg method for approximate minimization of a quadratic function
def dogleg(B, g, c, rho, M = None, M_inv = None, g_min = 1e-5):
    """Approximate minimum of quadr. fun `m: p -> c + g^T p + 1/2 * p^T B p` on ball of radius `rho` in metric `M`.
    
    Args:
        c: scalar. Constant term.
        g: covector in $R^n$. Linear term.
        B: matrix in $R^{n\times n}$. Quadratic term.
        rho: scalar. Radius of ball (wrt norm `M`).
        M: matrix in $R^{n\times n}$. Metric on $R^n$. Default is identity matrix.
        M_inv: inverse of `M`. Calculated if not provided.
        g_min: scalar. We check if the norm (wrt `M_inv`) of `g` larger than this.
        `
    Returns:
        p_min: point in $R^n$
        m(p_min): scalar
    """
    
    if M is None:
        M = np.eye(B.shape[0])
        
    if M_inv is None:
        M_inv = la.inv(M)
        
    assert g @ M_inv @ g > g_min**2, "Gradient too small."

    def m(p):
        return c + g @ p + 1/2 * p @ B @ p

    
    ## find the stationary point of `m`
    p_B = quadratic_stationary(B = B, g = g)
    m_B = m(p_B)
    
    ## find Cauchy point i.e. minimizer (in the trust-region) in the direction of maximal descent

    ## `u` = direction of the maximal descent of `f` wrt metric `M`
    u = - M_inv @ g

    ## normalize `u` wrt `M` 
    u /= norm(u, M)

    ## Cauchy point `x_U = x + p_U`
    p_U, m_U, t_U = minim_quadr_on_segment(
        segment = [np.zeros_like(g),  rho * u],
        B = B, g = g, c = c
    )
    assert m_U <= c, "m_U = {}, c = {}".format(m_U, c)
    

    if norm(p_B, M) <= rho and m_B < m_U:
        logger.debug("You can take global minimizer {} of our quadratic.".format(p_B)) 
        return p_B, m(p_B)
        

    if t_U < 1:
        if norm(p_B - p_U, M) <= 0.0001 * rho:
            logger.debug("An issue probably caused by numerical errors -- Cauchy point is too close to stationary point. We return Cauchy point.") 
            return p_U, m(p_U)

        ## Consider the line `p_U + t * (p_B - p_U)`. We want to find  its intersection with the 
        ## ball (in the metric `M`) with radius `rho`. 
        ## The intersection will be a `segment_2` given by two points in `R^n`.

        ## we restrict `norm**2 - rho**2` to the line
        alpha, beta, gamma = quadratic_restrict_to_line(
            a = p_U, 
            u = p_B - p_U,
            B = 2 * M,
            c = - rho **2
        )

        segment_2 = [p_U + t * (p_B - p_U) for t in solve_quadratic_1d(alpha, beta, gamma)]
        
#        print("Check quadratic solution: {}".format([alpha * t**2 + beta * t + gamma for t in solve_quadratic_1d(alpha, beta, gamma)]))
#        print("The norms of the endpoints of the segment: {}".format([norm(x, M) for x in segment_2]))

        ##
        p_s2, m_s2, blabla = minim_quadr_on_segment(
            segment = segment_2,
            B = B, g = g, c = c
        ) 

        if m_U < m_s2:
            logger.debug("Take Cauchy point {} with value {} is inside the region (t = {}).".format(p_U, m_U, t_U))
            logger.debug("It has smaller value than the minimizer {}, {} along the segment {}.".format(p_s2, m_s2, segment_2))
            return p_U, m_U
        else:
            logger.debug("Take point {} on the second segment with value {}.".format(p_s2, m_s2))
            return p_s2, m_s2
    else:
        logger.debug("The Cauchy point {}, with value {} is on the border of trust-region. You can use it.".format(p_U, m_U))
        return p_U, m_U
        
### Test       
#dogleg(
#    rho = 1,
#    c = 0,
#    g = np.array([-10, -1]),
#    B = np.diag([100, 1]),
#)

########################################
## Neat class for trust-region stepping

class Dogleg_Stepper:
    """
    Example usage:
    
    ```
    ## initialization of the optimization
    stepper = Dogleg_Stepper(
        x = x_start, 
        f_x = f(x_start), 
        df_x = df(x_start), 
        B = ddf(x_start)
    )
    
    ## optimization step (repeat this)
    x_sug = stepper.get_suggestion()
    stepper.update(f(x_sug), df(x_sug))
    print(stepper.state_description())
    ```
    
    Fields describing current state:
        x: point in $R^n$ -- current center of our trust region.
        f_x: scalar `f(x)`
        df_x: covector in $R^n$ equals `df(x)`
        B: current approximation of the Hessian 
        rho: radius of the trust-region
        n_iter: integer; number of iterations performed
        finished: boolean
        
    Fields holding parameters:
        M, M_inv: matrices for calculating norm of vectors and covectors        
        rho_max, minimal_decrease_ratio_to_take_step: scalar parameters
        minimal_df: scalar. We finish after the norm of the covector `df_x` (wrt `M_inv`) is smaller than this. 
    """
    
    def __init__(
        self,
        x, f_x, df_x, B,
        M = None,
        rho_start = 0.1,
        rho_max = 1,
        minimal_decrease_ratio_to_take_step = 0.1, # should be in `[0, 1/4)`
        minimal_df = 1e-5,
    ):
        self.x = x
        self.f_x = f_x
        self.df_x = df_x
        self.B = B
        self.M = M if M is not None else np.eye(x.shape[-1])
        self.M_inv = la.inv(self.M)
        self.rho = rho_start
        
        self.rho_max = rho_max
        self.minimal_decrease_ratio_to_take_step = minimal_decrease_ratio_to_take_step
        self.minimal_df = minimal_df
        
        self.n_iter = 0
        self.finished = False
        
        
        self.last_suggestion = None
        
    def get_suggestion(self):
        norm_of_grad = norm(self.df_x, self.M_inv)
        if norm_of_grad < self.minimal_df:
            self.last_suggestion = "Gradient small: norm(grad) = {} < {}".format(norm_of_grad, self.minimal_df)
            return self.x
            
        p, m_p = dogleg(
            rho = self.rho,
            c = self.f_x,
            g = self.df_x,
            B = self.B,
        )

        self.last_suggestion = SimpleNamespace(
            x = self.x + p, p = p, m_p = m_p
        )
        
        return self.x + p
    
    def update(self, f_x_sug, df_x_sug):
        ## retrieve last suggestion
        assert self.last_suggestion is not None, "No suggestion available. Call `get_suggestion`."
        
        if isinstance(self.last_suggestion, str):
            logger.info("We have finished: {}".format(self.last_suggestion))
            self.finished = True
            return
        
        p = self.last_suggestion.p
        m_p = self.last_suggestion.m_p
        
        ##
        x, f_x, df_x = self.x, self.f_x, self.df_x
        M = self.M
        
        ## calculate a `decrease_ratio` between the anticipated decrease (of quadratic approximation) and the real one
        decrease_ratio = (f_x_sug - f_x) / (m_p - f_x)
        logger.debug("Decrease of f is {}. Decrease of m is {}. Decrease_ratio = {}".format(
                -(f_x_sug - f_x), -(m_p - f_x), decrease_ratio, 
            )
        )
        
        ## adjust the radius `rho` of the trust-region
        if decrease_ratio < 1/4:
            self.rho *= 1/4
        else:
            if decrease_ratio > 3/4 and norm(p, M) >= 0.999 * self.rho:
                self.rho = min(2 * self.rho, self.rho_max)


        ## decide to make the step or not
        if decrease_ratio > self.minimal_decrease_ratio_to_take_step:
            self.B = BFGS_update_B(
                B0 = self.B, 
                x1 = x + p, x0 = x, 
                df_x1 = df_x_sug, df_x0 = df_x,
                check_positive = False
            )

            self.x = x + p
            self.f_x = f_x_sug 
            self.df_x = df_x_sug
        
        
        ##
        self.last_suggestion = None
        
        self.n_iter += 1
        
    def state_description(stepper):
        return "n_iter = {}; x = {}; f(x) = {}, df(x) = {}, rho = {}".format(
            stepper.n_iter, stepper.x, stepper.f_x, stepper.df_x, stepper.rho
        )
        

