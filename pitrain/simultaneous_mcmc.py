import tensorflow as tf
import numpy as np
import numpy.random as rnd

from pitf import ptf

from .misc import concat_dics

class MCMC_FF_Stepper:
    def __init__(self, ff, ff_losses, step_size = None, constraints = None, sess = None):
        """
        Args:
            ff: tensor of shape [batch_len, n] holding our variables.
            ff_losses: tensor of shape [batch_len]
            step_size: float number
            constraints: list of tensorflow tensors, each with shape [batch_len]
                The stepper does not go to places where any of the constraint is non-positive.
        """
        self.ff = ff
        self.ff_losses = ff_losses
        self.step_size = step_size
        self.constraints = constraints
        self.sess = sess
        
        
    def step(self, step_size = None, losses_current = None, feed_dict = None):
        """ Update `self.ff` using Metropolis-Hastings algorithm.
        
        We interpret `self.ff` with shape `[n, k]` as n independent k-dimensional variables.
        We basicaly do n independent Metropolis hasings updates on them.
        
            Args:
                step_size: float
                losses_current: numpy array of shape `[n]` holding current value of
                    the tensorflow node `self.ff_losses`. If not provided, we must 
                    evaluate it, slowing the process approximately 2 times.
                
            Returns:
                 numpy array `losses_new` of shape `n` holding the new value of `self.ff_losses`.
                
        """
        ff = self.ff
        ff_losses = self.ff_losses
        step_size = step_size if step_size is not None else self.step_size
        assert step_size is not None, "step_size is None"
        sess = self.sess if self.sess is not None else tf.get_default_session()

        ff_current = sess.run(ff, feed_dict)
        if losses_current is None:
            losses_current = sess.run(ff_losses, feed_dict)

        ## generate random proposal of the new step
        ff_suggest = rnd.normal(ff_current, scale = step_size)
        
        ## In the case the suggested step is outside the allowed area, 
        ## we make `ff_suggest := ff_current`. 
        ## (We do this before the calculation of `loss_suggest`, since the loss
        ## may not be defined outside of the allowed area).
        if self.constraints:
            constraints_val = sess.run(self.constraints, concat_dics(feed_dict, {ff: ff_suggest}))
            is_inside = np.all( [c > 0 for c in constraints_val], axis = 0)
            ff_suggest = np.where(is_inside[:, None], ff_suggest, ff_current)
        else:
            is_inside = True
        
        ## Calculate losses in the suggested place.
        losses_suggest = sess.run(ff_losses, concat_dics(feed_dict, {ff: ff_suggest}))

        ## following Metropolis-Hastings, we decide whether we make the step or not
        step_or_not = is_inside & rnd.binomial(
            n = 1,
            p = np.minimum(np.exp(losses_current - losses_suggest), 1)
        )
        
        step_or_not &= np.isfinite(losses_suggest)

        ##
        ff_new = np.where(np.array(step_or_not)[:, None], ff_suggest, ff_current)
        losses_new = np.where(step_or_not, losses_suggest, losses_current)

        ptf.assign_to(ff, ff_new)
        
        return losses_new
    
    def more_steps(self, n_steps, step_size = None, losses_current = None, feed_dict = None):
        """ Update `self.ff` using several steps of Metropolis-Hastings algorithm.
        
        See `self.step` description for details.        
        """
        for i in range(n_steps):
            losses_current = self.step(step_size, losses_current, feed_dict = feed_dict)
    
        return losses_current
    