import os
import numpy as np

from datasets import MNIST


class MNIST:
    def __init__(self, data_path):
        self.mnist_path = data_path
        self.train_data, self.train_labels = self.__load('train')
        self.test_data, self.test_labels = self.__load('t10k')

    def __load(self, kind='t10k'):
        data_path = f'{self.mnist_path}/{kind}-images.idx3-ubyte'
        label_path = f'{self.mnist_path}/{kind}-labels.idx1-ubyte'
        with open(label_path, 'rb') as f:
            f.read(8)
            labels = np.fromfile(f, dtype=np.uint8)
        with open(data_path, 'rb') as f:
            f.read(16)
            images = np.fromfile(f, dtype=np.uint8).reshape(len(labels), -1)
        return images, labels


class LinearSVM:
    def __init__(self):
        self.W = None

    def loss(self, X, y, reg):
        loss = 0.0
        dW = np.zeros(self.W.shape)
        scores = X.dot(self.W)
        num_train = X.shape[0]
        scores_y = scores[range(num_train), y].reshape(num_train, 1)
        margins = np.maximum(0, scores-scores_y+1)
        margins[range(num_train), y] = 0
        loss += np.sum(margins)/num_train
        loss += reg * np.sum(self.W ** 2)
        margins[margins > 0] = 1
        margins[range(num_train), y] = -np.sum(margins, axis=1)
        dW += np.dot(X.T, margins)
        dW += 2 * reg * self.W
        return loss, dW

    def train(self, X, y, learning_rate=1e-3, reg=1e-5, num_iters=100, batch_size=200, verbose=False):
        num_train, dim = X.shape
        num_classes = np.max(y) + 1
        if self.W is None:
            self.W = 0.001 * np.random.randn(dim, num_classes)

        loss_history = []
        for i in range(num_iters):
            X_batch = None
            y_batch = None
            idx = np.random.choice(num_train, batch_size)
            X_batch = X[idx]
            y_batch = y[idx]
            loss, grad = self.loss(X_batch, y_batch, reg)
            loss_history.append(loss)
            self.W = self.W - learning_rate * grad
            if verbose and i % 100 == 0:
                print('iteration %d / %d: loss %f' % (i, num_iters, loss))
        return loss_history

    def predict(self, X):
        y_pred = np.zeros(X.shape[0])
        scores = X.dot(self.W)
        y_pred = np.argmax(scores, axis=1)
        return y_pred


if __name__ == '__main__':
    dirname = os.path.dirname(__file__)
    mnist = MNIST(os.path.join(dirname, 'mnist'))
    svm = LinearSVM()
    svm.train(mnist.train_data, mnist.train_labels)
    y_pred = svm.predict(mnist.test_data)
    correct_num = np.sum(y_pred == mnist.test_labels)
    accuracy = 100 * correct_num/mnist.test_data.shape[0]
    print(f"分类准确率 {accuracy=:.2f}%")
