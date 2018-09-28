import tensorflow as tf

class BruteForceOptimization:
    def __init__(self, initializer, x_proposal, step_sequence):
        self.initializer = initializer
        self.x_proposal = x_proposal
        self.step_sequence = step_sequence
        
    @staticmethod
    def from_scratch(f, x, name = "brute_force_optimization"):
        with tf.name_scope(name):
            min_f = tf.Variable(f)
            argmin_f = tf.Variable(x)
            
            condition = tf.Variable(tf.ones_like(f, dtype = bool), name = "condition", dtype = tf.bool, validate_shape=False)
            condition.set_shape([None])    
            
            x_proposal = tf.placeholder(dtype = x.dtype, shape = x.shape[-1:], name = "x_proposal")
            
            initializer = tf.group(
                tf.assign(min_f, f, validate_shape = False, name = "init_min_f"),
                tf.assign(argmin_f, x, validate_shape = False, name = "init_argmin_f"),
                name = "initializer"
            )
            
            s1 = tf.group(
                tf.assign(x, x_proposal[None, :] * tf.ones_like(x), validate_shape= False),
                name = "s1"
            )
                
            
            s2 = tf.group(
                tf.assign(condition, f < min_f, validate_shape=False),
                name = "s2"
            )
            
            condition2 = tf.cast(
                tf.cast(condition[:, None], tf.int32) * tf.ones_like(x, tf.int32),
                bool
            )
            s3 = tf.group(
                tf.assign(min_f, tf.where(condition, f, min_f)),
                tf.assign(argmin_f, tf.where(condition2, x, argmin_f)),
                name = "s3"
            )
            
            x.eval()
            
            s4 = tf.group(
                tf.assign(x, argmin_f),
                name = "s4"
            )
            
            return BruteForceOptimization(
                    initializer= initializer,
                    x_proposal= x_proposal,
                    step_sequence = [s1, s2, s3, s4]
            )
            
    @staticmethod
    def find_in_graph(scope_prefix, graph = None):
        if graph is None:
            graph = tf.get_default_graph()
            
        if scope_prefix[-1] != "/":
            scope_prefix = scope_prefix + "/"
        
        return BruteForceOptimization(  
                initializer = graph.get_operation_by_name(scope_prefix + "initializer"),
                step_sequence = [
                    graph.get_operation_by_name(scope_prefix + "s" + str(i)) 
                    for i in [1, 2, 3, 4]
                    ],
                x_proposal = graph.get_tensor_by_name(scope_prefix + "x_proposal:0")
        )
        
    def run_initializer(self, sess = None):
        if sess is None:
            sess = tf.get_default_session()
        sess.run(self.initializer)
        
    def run_train_step(self, x_prop, sess = None):
        if sess is None:
            sess = tf.get_default_session()
        for s in self.step_sequence:
            sess.run(s, {self.x_proposal: x_prop})
    
