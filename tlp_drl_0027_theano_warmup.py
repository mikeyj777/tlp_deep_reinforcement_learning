import numpy as np
import theano
import theano.tensor as T
import tlp_drl_0025_cartpole_with_rbf_and_custom_gradient_descent as q_learning

class SGDRegressor:

    def __init__(self, D):
        w = np.random.randn(D) / np.sqrt(D)
        self.w = theano.shared(w)
        self.lr = 0.1
    
        X = T.matrix('X')
        Y = T.vector('Y')
        Y_hat = X.dot(self.w)
        delta = Y - Y_hat
        cost = delta.dot(delta)
        grad = T.grad(cost, self.w)
        updates = [(self.w, self.w - self.lr * grad)]

        self.train_op = theano.function(
            inputs = [X, Y],
            updates = updates
        )

        self.predict_op = theano.function(
            input=[X],
            outputs=Y_hat
        )
    
    def partial_fit(self, X, Y):
        self.train_op(X, Y)
    
    def predict(self, X):
        self.predict_op(X)

q_learning.main(show_plots=True)
