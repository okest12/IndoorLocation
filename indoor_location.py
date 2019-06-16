import numpy as np
import matplotlib.pylab as plt
from two_layer_net import TwoLayerNet

def load_train(train_file):
    train = np.loadtxt(train_file, delimiter=',', skiprows=1)
    size = train.shape[0]
    trainSize = 10000
    t_train = train[0:trainSize, 0:2]/200
    x_train = train[0:trainSize, 2:]/-130
    t_test = train[trainSize:size, 0:2]/200
    x_test = train[trainSize:size, 2:]/-130

    return (x_train, t_train), (x_test, t_test)


(x_train, t_train), (x_test, t_test) = load_train(r'F:\BaiduYunDownload\train.csv')
train_loss_list = []
iters_num = 1000
learning_rate = 0.001
network = TwoLayerNet(input_size=x_train.shape[1], hidden_size1=8, hidden_size2=4, output_size=t_train.shape[1])

for i in range(iters_num):
    loss = network.loss(x_train, t_train)
    print("round:%d, loss:%f" % (i, loss))
    train_loss_list.append(loss)

    grad = network.numerical_gradient(x_train, t_train)
    for key in ('W1', 'b1', 'W2', 'b2', 'W3', 'b3'):
        network.params[key] -= learning_rate * grad[key]


network.save_params()
print(200 * network.predict(x_train[800]))
print(200 * t_train[800])
print("loss:%f" % network.loss(x_train, t_train))
print("loss:%f" % network.loss(x_test, t_test))
plt.plot(np.arange(np.size(train_loss_list)), train_loss_list)
plt.show()
