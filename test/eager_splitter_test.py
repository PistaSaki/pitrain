import numpy as np
import tensorflow as tf
from pitrain.eager_splitter import Splitter2

def test_Splitter2():
    a = tf.Variable(np.zeros([2, 3]), dtype=tf.float32)
    b = tf.Variable(np.ones([5]), dtype=tf.float32)
    
    splitter = Splitter2([a, b])
    ##
    assert np.allclose(
        splitter.current_x(), 
        [0., 0., 0., 0., 0., 0., 1., 1., 1., 1., 1.]
    )
    ##
    splitter.assign_tensors(np.arange(11))
    assert np.allclose(
        a.numpy(),
        [[0., 1., 2.],
         [3., 4., 5.]]
    )
    assert np.allclose(b.numpy(), [ 6.,  7.,  8.,  9., 10.])

if __name__ == "__main__":
    test_Splitter2()
