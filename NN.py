
from data import load_CIFAR10
import numpy as np

# Nearest Neighbor


class NearestNeighbor:

    def __init__(self):
        self.Xtr_rows = None
        self.Xte_rows = None
        self.Xval_rows = None
        self.Ytr = None
        self.Yte = None
        self.Yval = None
        self.Yte_predict = None
        self.Yval_predict = None
    # end def

    @staticmethod
    def distance(x_1, x_2):
        return np.sum(np.abs(np.subtract(x_1, x_2)))
    #     return np.sqrt(np.sum(np.square()))
    # end def

    def train(self):
        Xtr, self.Ytr, Xte, self.Yte = load_CIFAR10("cifar10/")
        self.Xtr_rows = Xtr.reshape(Xtr.shape[0], 32 * 32 * 3)
        self.Xte_rows = Xte.reshape(Xte.shape[0], 32 * 32 * 3)
        print("Train finished.")
        return
    # end def

    def nearest_neighbor(self, x):
        min_y = self.Yte[0]
        min_distance = self.distance(x, self.Xtr_rows[0])
        for i, x_tr in enumerate(self.Xtr_rows):
            dist = self.distance(x, x_tr)
            if dist < min_distance:
                min_y = self.Ytr[i]
                min_distance = dist
        # end for
        return min_y
    # end def

    def predict(self, count=10000):
        assert count <= 10000
        self.Yte_predict = np.array([])
        for i, x_te in enumerate(self.Xte_rows[:count]):
            y_te_predict = self.nearest_neighbor(x_te)
            self.Yte_predict = np.append(self.Yte_predict, y_te_predict)
            if (i+1) % 100 == 0:
                print("\tPredicted {} images".format(i+1))
                print(self.Yte_predict[i-99:i+1])
        # end for
        print("Predict finished")
        return self.Yte_predict
    # end def

    def accuracy(self, count=10000):
        return np.mean(self.Yte_predict[:count] == self.Yte[:count])


if __name__ == '__main__':
    nn = NearestNeighbor()
    nn.train()
    nn.predict(1000)
    print(nn.accuracy(1000))

    # Xtr, Ytr, Xte, Yte = load_CIFAR10("cifar10/")
    # Xtr_rows = Xtr.reshape(Xtr.shape[0], 32 * 32 * 3)
    # Xte_rows = Xte.reshape(Xte.shape[0], 32 * 32 * 3)


