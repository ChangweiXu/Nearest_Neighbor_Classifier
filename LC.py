
from data import load_CIFAR10
import numpy as np


class LinearClassifier:

    def __init__(self):
        temp = [0 for j in range(32*32*3)]
        self.W = [np.array(temp) for i in range(10)]
        print("Initialized")

    def train(self, Xtr, Ytr):
        # TODO: optimization algorithm
        pass
        print("Train finished")

    def predict_one(self, X_test):
        return np.argmax(np.dot(self.W, X_test))

    def predict(self, Xte):
        Yte_predict = []
        for X_test in Xte:
            Yte_predict.append(self.predict_one(X_test))
        return Yte_predict


if __name__ == '__main__':
    Xtr, Ytr, Xte, Yte = load_CIFAR10("cifar10/")
    Xtr_rows = Xtr.reshape(Xtr.shape[0], 32 * 32 * 3)
    Xte_rows = Xte.reshape(Xte.shape[0], 32 * 32 * 3)

    lc = LinearClassifier()
    lc.train(Xtr_rows, Ytr)
    test_predict = lc.predict(Xte_rows)
    print(test_predict)
# end main
