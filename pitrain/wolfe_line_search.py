import numpy as np
import matplotlib.pyplot as pl
from time import sleep

from types import SimpleNamespace
from pandas import DataFrame

from .misc import BFGS_update_B

class Rememberer:
    def __init__(self, f):
        self.f = f
        self.history = DataFrame()
        
    def __call__(self, x):
        val = self.f(x)
        self.history = self.history.append(
            other = {
                "x": x,
                "f": val,
            },
            ignore_index = True
        )
        
        return val

class WolfeLineSearchException(Exception):
    pass

def wolfe_linear_search(
    phi, d_phi,
    alpha_init,
    c1 = 0.5,
    c2 = 0.7,
    phi_0 = None, d_phi_0 = None,
    alpha_max = None,
    
    bracketing_augment_method = lambda alpha, **kwargs: 2 * alpha,
    

    max_noof_bracketing_steps = 10,
    max_noof_zooming_steps = 10,
    
    printing = True, ax = None,
    plotting_params = dict()
):
    """
    0 < c1 < c2 < 1
    
    `plotting_params` is a dictionary. It can comtain:
        `plot_fun` : Boolean whether the function should be plotted at the start.
        `n`: if it is plotted, how many points are calculated
        `sleep_time`: how many seconds we should wait after each step. Defaults 0.
        `alpha_max`: f is originally plotted on [0, alpha_max]
    """

    if phi_0 is None:
        phi_0 = phi(0)
        
    if d_phi_0 is None:
        d_phi_0 = d_phi(0)
    

    ## initialize plotting
    plotting = ax is not None
    if plotting:
        phi = Rememberer(phi)
        pl_a_max = plotting_params.get("alpha_max", 5 * alpha_init)
            
        if plotting_params.get("plot_fun", False):
            ## we evaluate `phi` on a grid 
            ## so the calculated values will be written into `phi.history`.
            for x in np.linspace(0, pl_a_max, plotting_params.get("n", 10)):
                phi(x)
        else:
            phi.history = phi.history.append({"x": 0, "f": phi_0}, ignore_index = True)
    
        fun_plot = pl.plot(phi.history.x, phi.history.f)[0]
        
            
        # plot c1, c2 lines
        ax.autoscale(False)
        pl.plot([0, pl_a_max],[phi_0, phi_0 + c1 * d_phi_0 * pl_a_max],)
        pl.plot([0, pl_a_max],[phi_0, phi_0 + c2 * d_phi_0 * pl_a_max],)
        ax.autoscale(True)

        ## prepare the dots depicting current state of our algorithm
        dots = ax.scatter(0, phi_0)

    def update_plot(new_dots = []):
        if plotting:
            dots.set_offsets(new_dots)
            
            vvv = phi.history.sort_values("x")
            fun_plot.set_xdata(vvv.x)
            fun_plot.set_ydata(vvv.f)
            
            # recompute the ax.dataLim
            ax.relim()
            # update ax.viewLim using the new dataLim
            ax.autoscale_view()
            
            # redraw
            ax.figure.canvas.draw()
            
            if "sleep_time" in plotting_params:
                sleep(plotting_params["sleep_time"])
            

    ## Initialize bracketing phase
    a_prev = 0
    phi_a_prev = phi_0

    a = alpha_init
    phi_a = phi(a)

    
    if plotting:
        # update plot
        update_plot([[a, phi_a], [a_prev, phi_a_prev]])
        

    ###################
    ## Bracketing phase

    for i in range(max_noof_bracketing_steps):
        print("New bracketing step with a_prev={}; a={}".format(a_prev, a))
        # If the 'sufficient descent' fails or we are already increasing (from `a_prev`)
        # then this means that we have increased `a` far enough and we can start zooming.
        if (phi_a > phi_0 + c1 * a * d_phi_0) or (phi_a >= phi_a_prev):
            a_lo, a_hi = a_prev, a
            phi_a_lo, phi_a_hi = phi_a_prev, phi_a

            if printing:
                if (phi_a > phi_0 + c1 * a * d_phi_0):
                    print("Sufficient descent fails.")
                if (phi_a >= phi_a_prev):
                    print("phi_a >= phi_a_prev")
                print("You can start zooming with a_lo={}, a_hi={}".format(a_lo, a_hi))
            break

        d_phi_a = d_phi(a)

        # If 'curvature condition' is satisfied then we are done since we already know 
        # that a satisfies the 'sufficient descent'.
        if np.abs(d_phi_a) <= c2 * np.abs(d_phi_0):
            if printing:
                print("You are done and you can use alpha ={}".format(a))
            if plotting:
                ## plot the final alpha
                ax.scatter(a, phi_a, s = 40, c = "r")
                ax.figure.canvas.draw()

            return a

        # If `phi` is increasing in `a`, we can start zooming.
        if(d_phi_a >= 0):
            a_lo, a_hi = a, a_prev
            phi_a_lo, phi_a_hi = phi_a, phi_a_prev
            if printing:
                print("d_phi_a >= 0;")
                print("You can start zooming with a_lo={}, a_hi={}".format(a_lo, a_hi))
            break

        ## If none of the previous happens, we must start everything with larger `a`:
        # choose new a in (a, alpha_max):
        if a == alpha_max:
            raise WolfeLineSearchException("We reached a == alpha_max in bracketing phase. Can't augment it any more.")
            
        a_new = bracketing_augment_method(alpha = a)
        
        if alpha_max is not None:
            a_new = min(a_new, alpha_max)

        a_prev, phi_a_prev = a, phi_a
        a, phi_a = a_new, phi(a_new)

        if printing:
            print("Our a={} seems not to be large enough. Try with larger a ={}".format(a_prev, a))
        if plotting:
            update_plot([[a, phi_a], [a_prev, phi_a_prev]])
            
    else:
        raise WolfeLineSearchException("Too many steps in bracketing phase.")
        
    

    ################
    ## Zooming phase
    # In the whole process:
    # a) the interval bounded by `a_lo`, `a_hi` contains step-lengths that satisfy Wolfe.
    #   (Warning: `a_hi` can be smaller than `a_lo`.)
    # b) `a_lo` satisfies the 'sufficient decrease' and
    #   `phi(a_lo) = min {phi(a) that we calculated so far and a satisfied sufficient decrease}`.
    # c) in particular `a_hi` either does not satisfy 'sufficient decrease' or 
    #    it does and in that case it phi(a_hi) > phi(a_lo)
    # d) $\phi'(a_{lo})\cdot(a_{hi} - a_{lo})$ 
    # This means that phi is decreasing in `a_lo` in the direction towards `a_hi`.


    
    for j in range(max_noof_zooming_steps):

        print("New zooming step with a_lo = {}; a_hi = {}".format(a_lo, a_hi))
        # update plot
        if plotting:
            update_plot([[a_lo, phi_a_lo], [a_hi, phi_a_hi]])
            

        # choose `a` between `a_lo` and `a_hi`
        a = (a_lo + a_hi) / 2
        phi_a = phi(a)

        # If 'sufficient decrease' at `a` fails, or we get larger value than at `a_lo`
        # then we shrink the interval: keep `a_lo` and take `a_hi = a`.
        if (phi_a > phi_0 + c1 * a * d_phi_0) or (phi_a >= phi_a_lo):
            a_hi = a
            phi_a_hi = phi_a
        else:
            # Now we know that 'sufficient decrease' holds at `a`.
            # Check the 'curvature condition':
            d_phi_a = d_phi(a)    
            if abs(d_phi_a) <= c2 * abs(d_phi_0):
                print("You are done and you can use alpha ={}".format(a))
                break

            # If we got here, we know that 'curvature condition' failed so we must 
            # update our interval. 
            # We also know that `a` satisfies the 'sufficient decrease' and `phi_a < phi_lo'.
            # Thus (according to (b)) in the next step, `a` must become the new `a_lo`.
            # Therefore new `a_lo, a_hi`  will be either `a, a_hi` or `a, a_lo`.
            # We choose so that
            # phi will be decreasing at `a_lo` in the direction towards `a_hi`.
            if d_phi_a * (a_hi - a_lo) >= 0:
                a_hi = a_lo
                phi_a_hi = phi_a_lo

            a_lo = a
            phi_a_lo = phi_a
    else:
        raise WolfeLineSearchException("Too many steps in zooming phase.")
        
        
    if plotting:
        ## plot the final alpha
        ax.scatter(a, phi_a, s = 40, c = "r")
        ax.figure.canvas.draw()

    return a
    
    
class Wolfe_Line_Search:
    def __init__(self,
        f, df, 
        c1 = 0.01,
        c2 = 0.4,
        
        bracketing_augment_method = lambda alpha, **kwargs: 2 * alpha,

        max_noof_bracketing_steps = 10,
        max_noof_zooming_steps = 10,

        printing = True, 
        ax = None,
        plotting_params = dict()
    ):
        """
        0 < c1 < c2 < 1

        `plotting_params` is a dictionary. It can comtain:
            `plot_fun` : Boolean whether the function should be plotted at the start.
            `n`: if it is plotted, how many points are calculated
            `sleep_time`: how many seconds we should wait after each step. Defaults 0.
            `alpha_max`: f is originally plotted on [0, alpha_max]
        """
        self.f = f
        self.df = df
        
        self.c1 = c1
        self.c2 = c2
        
        self.bracketing_augment_method = bracketing_augment_method
        
        self.max_noof_bracketing_steps = max_noof_bracketing_steps
        self.max_noof_zooming_steps = max_noof_zooming_steps
        
        self.printing = printing
        self.ax = ax
        self.plotting_params = plotting_params
    
    def search(self,
        x, p, 
        f_x = None, df_x = None, 
        alpha_init = 1,
        alpha_max = None,
    ):
        """
        Returns an object with fields: `alpha, x_new,f_x_new, df_x_new`
        """
        ## We want to memorize the evaluations, so that we can return the values at the new `x`
        f_remember = Rememberer(self.f)
        df_remember = Rememberer(self.df)
        
        ## restrict `f` and `df` to line {x + a * p| a > 0}
        def phi(a):
            return f_remember(x + a * p)

        d_phi = lambda a: p @ df_remember(x + a * p)
        

        ## do the search
        if self.ax is not None:
            self.ax.clear()
            
        alpha = wolfe_linear_search(
            phi = phi, 
            d_phi = d_phi,
            alpha_init = alpha_init,
            c1 = self.c1,
            c2 = self.c2,
            
            phi_0 = f_x, 
            d_phi_0 = p @ df_x if df_x is not None else None,
            
            alpha_max = alpha_max,

            bracketing_augment_method = self.bracketing_augment_method,

            max_noof_bracketing_steps = self.max_noof_bracketing_steps,
            max_noof_zooming_steps = self.max_noof_zooming_steps,

            printing = self.printing, 
            ax = self.ax,
            plotting_params = self.plotting_params
        )

        ## update `x`
        x_new = x + alpha * p

        ## retrieve value f_x from history or calculate it
        if all(f_remember.history.x.iloc[-1] == x_new):
            f_x_new = f_remember.history.f.iloc[-1]
        else:
            raise WolfeLineSearchException("I think the value of `f` at `x` should have been calkulated in Wolfe search.")

        ## retrieve value df_x from history or calculate it
        if all(df_remember.history.x.iloc[-1] == x_new):
            df_x_new = df_remember.history.f.iloc[-1]
        else:
            raise WolfeLineSearchException("I think the value of `df` at `x` should have been calkulated in Wolfe search.")
            
        ## 
        return SimpleNamespace(
            alpha = alpha,
            x_new = x_new,
            f_x_new = f_x_new,
            df_x_new = df_x_new
        )


