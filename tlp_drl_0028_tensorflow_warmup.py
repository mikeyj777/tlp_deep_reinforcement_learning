import numpy as np
import tensorflow as tf
import tlp_drl_0025_cartpole_with_rbf_and_custom_gradient_descent as q_learning

class SGDRegressor:

    def __init__(self, D):
        lr = 0.1

        self.w = tf.Variable(tf.random_normal(shape=(D,1)), name = 'w')
        self.X = tf.placeholder(tf.float32, shape = (None, D), name = 'X')
        self.Y = tf.placeholder(tf.float32, shape = (None,), name = 'Y')

        Y_hat = tf.reshape( tf.matmul(self.X, self.w), [-1])
        delta = self.Y - Y_hat
        cost = tf.reduce_sum(delta * delta)

        self.train_op = tf.train.GradientDescentOptimizer(lr).minimize(cost)
        self.predict_op = Y_hat

        init = tf.global_variables_initializer()
        self.session = tf.InteractiveSession()

    def partial_fit(self, X, Y):
        self.session.run(self.train_op, feed_dict={self.X: X, self.Y:Y})
    
    def predict(self, X):
        return self.session.run(self.predict_op, feed_dict={self.X: X})
    
q_learning.main(show_plots=True)