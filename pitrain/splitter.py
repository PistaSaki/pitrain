from collections import  OrderedDict

class Splitter:
    """Class for splitting one numpy vector into many tensors and joining back.
    
    It is used if you have a tensorflow function of several tensor variables and the optimalization algorithm was designed
    to work with real functions accepting one numpy vector as an argument.
    """
    def __init__(self, tensors):
        """
        Args:
            tensors: list of tensorflow tensors woth well defined shape.
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
    
    def split_dic(self, x):
        """Split numpy vector `x` into tensorflow feed_dict for `self.tensors`.
        
        Args:
            x: numpy vector
            
        Returns:
            Dictionary whose keys are `self.tensors` and values are numpy arrays containing
            the elements from `x`.
        """
        return OrderedDict( 
            zip(self.tensors, self.split(x)) 
        )
    
    def join(self, lst):
        return np.concatenate( [tv.flatten() for tv in lst])
    
    def join_dic(self, dic):
        return self.join([dic[t] for t in self.tensors])
    
    def current_x(self, sess = None):
        """Evaluate `self.tensors` and join them into one numpy vector."""
        if sess is None:
            sess = tf.get_default_session()
        return self.join(sess.run(self.tensors))
    
    def assign_tensors(self, x):
        for var, val in self.split_dic(x).items():
            ptf.assign_to(var, val)
    
        
    