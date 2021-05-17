"""
作业9：
    试着提取MNIST手写数字数据库（https://yann.lecun.com/exdb/mnist/）的一些特征（取部分数据集做训练，如每个数字10张图片做训练），
    并用某种分类算法对测试数据（10张不同数字图片做测试）进行分类，检查你所获得的分类准确率。
    https://www.icourse163.org/spoc/learn/FZU-1462424162?tid=1463209447#/learn/hw?id=1237735420
参考：
    https://zhuanlan.zhihu.com/p/35856929
"""
import os
import numpy as np

from datasets import MNIST


class LinearSVM:
    def __init__(self):
        self.W = None

    def loss(self, X, y, reg):
        loss = 0.0
        dW = np.zeros(self.W.shape)
        scores = X.dot(self.W)
        num_train = X.shape[0]

        scores_y = scores[range(num_train), y].reshape(num_train, 1)
        # 所有数据的正确类得分
        # print("scores_y", scores_y.shape)  # scores_y (500, 1)
        margins = np.maximum(0, scores-scores_y+1)
        # print("margins", margins.shape)  # margins (500, 10)
        margins[range(num_train), y] = 0
        # 因为正确分类不需要参与loss计算，此处置0方便后面sum
        loss += np.sum(margins)/num_train
        loss += reg * np.sum(self.W ** 2)

        margins[margins > 0] = 1
        # 重复利用上面的margins，而它在上面已经将所有<=0的都设为了0
        margins[range(num_train), y] = -np.sum(margins, axis=1)
        # 按行计算不正确分类的个数总和，并赋给正确分类所在位置
        dW += np.dot(X.T, margins)
        # X.T：(D, N)， margins：(N, C), C=10
        dW += 2 * reg * self.W
        return loss, dW

    def train(self, X, y, learning_rate=1e-3, reg=1e-5, num_iters=100, batch_size=200, verbose=False):
        num_train, dim = X.shape
        num_classes = np.max(y) + 1
        # 类别从0-n-1编号，取数据里最大y值加1即为总类别数
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
        # 算出得分(N, C) 然后每行中最大的那个就是所预测的分类
        return y_pred


if __name__ == '__main__':
    dirname = os.path.dirname(__file__)
    mnist = MNIST(os.path.join(dirname, 'mnist'))
    # mnist.show()
    svm = LinearSVM()
    svm.train(mnist.train_data, mnist.train_labels)
    y_pred = svm.predict(mnist.test_data)
    correct_num = np.sum(y_pred == mnist.test_labels)
    accuracy = 100 * correct_num/mnist.test_data.shape[0]
    print(f"分类准确率 {accuracy=:.2f}%")
