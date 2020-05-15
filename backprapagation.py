
# created at 2020.05.15 by Gaofeng

# 通过面向对象机制,分离了层与层之间的前向传播和反向传播

# %%
import numpy as np
from random import shuffle
from mnist import *

def sigmoid_de(x):
    return sigmoid(x) * (1.0 - sigmoid(x))
def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

class net:

    # 初始化权重
    def __init__(self, layers):
        self.layers_num = len(layers)
        self.weights = [ np.random.randn(j, i) 
            for i, j in zip(layers[:-1], layers[1:]) ]
        self.baises = [ np.random.randn(j, 1)
            for j in layers[1:] ]

    # 随机梯度下降
    def SGD(self, train_data, learning_rate, batch_size, epoches, test_data=None):
        for epoch in range(epoches):
            shuffle(train_data)
            batches = [ train_data[i:i + batch_size] \
                for i in range(0, len(train_data), batch_size) ]
            for batch in batches:
                self.update_batch(batch, learning_rate)
            if test_data:
                print("epoch {}: {} / {}".
                    format(epoch, self.evaluate(test_data), len(test_data)))
            else:
                print("epoch {}".format(epoch))   

    # 估计测试集上的误差
    def evaluate(self, data):
        sum = 0
        for x, y in data:
            for w, b in zip(self.weights, self.baises):
                x = sigmoid(np.dot(w, x) + b) 
            sum += np.sum(np.power(y - x, 2))
        return sum

    # 根据一个batch_size上的数据更新权重
    def update_batch(self, batch, learning_rate):
        nabla_w = [ np.zeros(i.shape) for i in self.weights ]
        nabla_b = [ np.zeros(i.shape) for i in self.baises ]
        for xy in batch:
            nabla_w_, nabla_b_= self.back(xy[0], xy[1])
            nabla_w = [ i + j for i, j in zip(nabla_w, nabla_w_) ]
            nabla_b = [ i + j for i, j in zip(nabla_b, nabla_b_) ]
        self.weights = [ i - j * learning_rate / batch_size 
            for i, j in zip(self.weights, nabla_w) ]
        self.baises = [ i - j * learning_rate / batch_size 
            for i, j in zip(self.baises, nabla_b) ]

    # 对一个图片进行反向传播(!不是一个batch_size)
    def back(self, x, y):
        nabla_w = [ np.zeros(i.shape) for i in self.weights ]
        nabla_b = [ np.zeros(i.shape) for i in self.baises ]
        zs = []
        activation = x
        activations = [x]

        for w, b in zip(self.weights, self.baises):
            z = np.dot(w, activation) + b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)

        for index in range(1, self.layers_num) :
            index = -index
            if index == -1:
                delta = self.cost_de(y, activations[-1]) * sigmoid_de(zs[-1])
            else:
                delta = np.dot(self.weights[index + 1].T, delta) * sigmoid_de(zs[index])
            nabla_w[index] = np.dot(delta, activations[index - 1].T)
            nabla_b[index] = delta
        return nabla_w, nabla_b
    
    # 损失函数的导数
    def cost_de(self, y, y_):
        return y_ - y


# %%

if __name__ == "__main__":

    # 读取数据集
    x_train, y_train, x_valid, y_valid = read_mnist()
    train_data = [ (x, y) for x, y in zip(x_train, y_train) ]
    train_data = train_data[:20000]
    test_data = [ (x, y) for x, y in zip(x_valid, y_valid) ]
    test_data = test_data[:2000]

    # 参数
    batch_size = 24
    learning_rate = 1
    epoches = 5
    
    # 练
    model = net([784, 30, 10])
    model.SGD(train_data, learning_rate, batch_size, epoches, test_data)

