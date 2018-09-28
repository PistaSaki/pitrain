import tensorflow as tf

class GragientDescentWithBacktracking:
    def __init__(self, initializer, step_sequence):
        self.initializer = initializer
        self.step_sequence = step_sequence
        
    def run_initializer(self, sess = None):
        if sess is None:
            sess = tf.get_default_session()
        sess.run(self.initializer)
        
    def run_train_step(self, sess = None):
        if sess is None:
            sess = tf.get_default_session()
        for s in self.step_sequence:
            sess.run(s)
            
    @staticmethod
    def find_in_graph(scope_prefix, graph = None):
        if graph is None:
            graph = tf.get_default_graph()
            
        if scope_prefix[-1] != "/":
            scope_prefix = scope_prefix + "/"
        
        return GragientDescentWithBacktracking(  
                initializer = graph.get_operation_by_name(scope_prefix + "initializer"),
                step_sequence = [
                    graph.get_operation_by_name(scope_prefix + "s" + str(i)) 
                    for i in range(1, 7)
                ],
        )
    
    
    @staticmethod
    def from_scratch(x, f, df, name = "gradient_descent_with_backtracking", 
        initializer_kwargs = {"initial_step_length": 1},
        train_step_kwargs =  {"c1": 0.5, "rho": 0.5},                    
    ):
        """
        Args:
            x: tensorflow variable of shape `[batch_len, n]`
            f: tensor of shape `[batch_len]`
            initializer_kwargs: dictionary
            train_step_kwargs: dictionary            
        """
        def construct_initializer(
            initial_step_length,
            minimal_grad_length = 1e-5,
        ):
            """Return initializer which sets such alpha to get specified step length.
            
            Namely alpha is set by the formula:
                `alpha = initial_step_length / max(minimal_grad_length, norm(df))`        
            
            Args:
                initial_step_length: scalar. Step length we want to achieve.
                minimal_grad_length: scalar
                
            """
            return tf.group(
                *[
                    tf.assign(
                        var, val, name = name + "_initializer", 
                        validate_shape=False
                    )
                    for var, val, name in [
                        [x_c, x, "x_c"],
                        [f_c, f, "f_c"],
                        [df_c, df, "df_c"],
            
                        [x_p, x, "x_p"],
                        [f_p, f, "f_p"],
                        [df_p, df, "df_p"],        
                    ]
                ], 
                tf.assign(
                    alpha,
                    initial_step_length / tf.maximum(minimal_grad_length, tf.norm(df, axis = -1)),
                    name = "alpha_initializer", 
                    validate_shape = False
                ),
                tf.assign(
                    condition, tf.cast(tf.ones_like(f), tf.bool), 
                    validate_shape=False),
                name = "initializer"
            )
                
        def construct_step_sequence(c1 = 0.5, rho =  0.5):
            """
            Args:
                c1: scalar. sufficient_descent_const
                rho:scalar. step_contraction_factor
            """
            ### Step 1
            ## $$x_p := x_c - alphacdot df_c$$
            s1 = tf.group(
                tf.assign(x_p, x_c - alpha[:, None] * df_c), 
                name = "s1"
            )
            
            ### Step 2 and 3
            ## $$f_p := f(x_p)$$
            ## $$df_p := df(x_p)$$
            
            s2 = tf.group(
                tf.assign(x, x_p + 1 - 1), 
                name = "s2"
            )
            s3 = tf.group(
                tf.assign(f_p, f + 1 - 1, name = "evaluate_f_p"),
                tf.assign(df_p, df + 1 - 1, name = "evaluate_df_p"),
                name = "s3"
            )
            
            ### Step 4 and 5
            ## Sufficient descent condition is:
            ## $$f_p leq f_c + c_1 (x_p - x_c)cdot df_c $$
            ## If it is satisfied then:
            ## $$x_c := x_p; quad f_c := f_p; quad df_c := df_p; quad alpha := alpha; quad  $$
            ## otherwise
            ## $$x_c := x_c; quad f_c := f_c; quad df_c := df_c; quad alpha := \rhocdot alpha; quad  $$
            
            s4 = tf.group(
                tf.assign(
                    condition,
                    f_p <= f_c + c1 * tf.einsum("bi,bi->b", x_p - x_c, df_c),
                    name = "evaluate_condition",
                    validate_shape = True,
                ),
                name = "s4"
            )
            
            condition2 = tf.cast(
                tf.cast(condition[:, None], tf.int32) * tf.ones_like(x, tf.int32),
                bool
            )
    
            s5 = tf.group(
                    tf.assign(x_c, tf.where(condition2, x_p, x_c)),
                    tf.assign(f_c, tf.where(condition, f_p, f_c)),
                    tf.assign(df_c, tf.where(condition2, df_p, df_c)),
                    tf.assign(alpha, tf.where(condition, alpha, rho * alpha)),
                    name = "s5"
                )
            
            s6 = tf.group(tf.assign(x, x_c), name = "s6")
            
            return [s1, s2, s3, s4, s5, s6]
          
        with tf.name_scope(name):
            
            floatX = f.dtype
            
            # define vars we need
            
            x_c = tf.Variable(x, name = "x_c", dtype = floatX, validate_shape=False)
            x_c.set_shape([None, None])
        
            f_c = tf.Variable(f, name = "f_c", dtype = floatX, validate_shape=False)
            f_c.set_shape([None])
        
            df_c = tf.Variable(x, name = "df_c", dtype = floatX, validate_shape=False)
            df_c.set_shape([None, None])
        
            x_p = tf.Variable(x, name = "x_p", dtype = floatX, validate_shape=False)
            x_p.set_shape([None, None])
        
            f_p = tf.Variable(f, name = "f_p", dtype = floatX, validate_shape=False)
            f_p.set_shape([None])
        
            df_p = tf.Variable(x, name = "df_p", dtype = floatX, validate_shape=False)
            df_p.set_shape([None, None])
        
            condition = tf.Variable(tf.ones_like(f, dtype = bool), name = "condition", dtype = tf.bool, validate_shape=False)
            condition.set_shape([None])
        
            alpha = tf.Variable(tf.ones_like(f), name = "alpha", dtype = floatX, validate_shape=False)
            alpha.set_shape([None])
        
    
            return GragientDescentWithBacktracking(
                initializer=construct_initializer(**initializer_kwargs),
                step_sequence=construct_step_sequence(**train_step_kwargs)
            )
        
      
      