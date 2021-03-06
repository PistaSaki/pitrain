from collections import  OrderedDict
import numpy as np
import tensorflow as tf

class Splitter2:
    """Class for splitting one numpy vector into many tensors and joining back.
    
    It is used if you have a tensorflow function of several tensor variables and the optimalization algorithm was designed
    to work with real functions accepting one numpy vector as an argument.
    
    Thi class is meant to replace `splitter.Splitter` in new tensorflow 2.
    """
    def __init__(self, tensors):
        """
        Args:
            tensors: list of tensorflow tensors with well defined shape.
        """
        self.tensors = tensors
        
        
    def split(self, x):
        """Split numpy vector `x` into list of multiple arrays of shapes as `self.tensors`.
        
        Args:
            x: numpy vector
            
        Returns:
            list of numpy arrays
        """
        sizes = [int(np.prod(t.shape)) for t in self.tensors]
        assert sum(sizes) == len(x)
        
        split_indices = np.cumsum(sizes)[:-1]
        
        return [
            xt.reshape(t.shape)
            for t, xt in zip(self.tensors, np.split(x, split_indices))
        ]
    
    
    def join(self, lst):
        flattenned = [tf.reshape(x, [-1]) for x in lst]
        return tf.concat( flattenned, axis=0).numpy()
    
    def current_x(self):
        """Evaluate `self.tensors` and join them into one numpy vector."""
        return self.join(self.tensors)
    
    def assign_tensors(self, x):
        for var, val in zip(self.tensors, self.split(x)):
            var.assign(val)
    