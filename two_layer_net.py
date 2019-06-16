import os
import pickle

import numpy as np


def relu(x):
    #return (abs(x) + x) / 2
    return x


def min_squared_error(y, t):
    return np.sum(np.sqrt((y[:, 0] - t[:, 0]) ** 2 + (y[:, 1] - t[:, 1]) ** 2)) / y.shape[0]
    #return np.sqrt((np.sum((y - t)**2)) / y.shape[0])


def numerical_gradient(f, x):
    h = 1e-4  # 0.0001
    grad = np.zeros_like(x)

    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:
        idx = it.multi_index
        tmp_val = x[idx]
        x[idx] = float(tmp_val) + h
        fxh1 = f(x)  # f(x+h)

        x[idx] = tmp_val - h
        fxh2 = f(x)  # f(x-h)
        grad[idx] = (fxh1 - fxh2) / (2 * h)

        x[idx] = tmp_val
        it.iternext()

    return grad


class TwoLayerNet:

    def __init__(self, input_size, hidden_size1, hidden_size2, output_size, weight_init_std=0.1):
        sample_weight_file = r'sample_weight.pkl'
        if os.path.exists(sample_weight_file):
            self.params = pickle.load(open('sample_weight.pkl', 'rb'))
        else:
            self.params = dict(W1=weight_init_std * np.random.randn(input_size, hidden_size1),
                               b1=weight_init_std * np.ones(hidden_size1),
                               W2=weight_init_std * np.random.randn(hidden_size1, hidden_size2),
                               b2=weight_init_std * np.ones(hidden_size2),
                               W3=weight_init_std * np.random.randn(hidden_size2, output_size),
                               b3=weight_init_std * np.ones(output_size))

    def save_params(self):
        pickle.dump(self.params, open('sample_weight.pkl', 'wb'))

    def predict(self, x):
        w1, w2, w3 = self.params['W1'], self.params['W2'], self.params['W3']
        b1, b2, b3 = self.params['b1'], self.params['b2'], self.params['b3']

        a1 = np.dot(x, w1) + b1
        z1 = relu(a1)
        a2 = np.dot(z1, w2) + b2
        z2 = relu(a2)
        a3 = np.dot(z2, w3) + b3
        y = relu(a3)
        return y

    def loss(self, x, t):
        y = self.predict(x)

        return min_squared_error(y, t)

    def numerical_gradient(self, x, t):
        loss_W = lambda W: self.loss(x, t)

        grads = {}
        for key in ('W1', 'b1', 'W2', 'b2', 'W3', 'b3'):
            grads[key] = numerical_gradient(loss_W, self.params[key])

        return grads

    def gradient(self, x, t):
        W1, W2 = self.params['W1'], self.params['W2']
        b1, b2 = self.params['b1'], self.params['b2']
        grads = {}

        batch_num = x.shape[0]

        # forward
        a1 = np.dot(x, W1) + b1
        z1 = sigmoid(a1)
        a2 = np.dot(z1, W2) + b2
        y = softmax(a2)

        # backward
        dy = (y - t) / batch_num
        grads['W2'] = np.dot(z1.T, dy)
        grads['b2'] = np.sum(dy, axis=0)

        da1 = np.dot(dy, W2.T)
        dz1 = sigmoid_grad(a1) * da1
        grads['W1'] = np.dot(x.T, dz1)
        grads['b1'] = np.sum(dz1, axis=0)

        return grads


def test1():
    y = np.array([[3, 4], [5, 6], [3, 7]])
    t = np.array([[5, 6], [7, 8], [5, 4]])
    print(min_squared_error(y, t))

