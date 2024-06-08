import numpy as np
import matplotlib.pyplot as plt
import pickle

class EarlyStopper:
    def __init__(self):
        self.train_losses = []
        self.test_losses = []

        self.best_test_loss = 100

        self.IsConverging = 0
        self.NotConverging = 0
        self.ConvergingMaxIter = 5

        self.MinDiff = -10 ** -3
        self.batch = 2

        self.epoch = -1
        self.MaxIter = -1

        self.PlotTitle = 'Loss Per Epoch'

    def Loss(self, train_loss, test_loss):
        self.epoch += 1
        self.train_losses += [train_loss]
        self.test_losses += [test_loss]

        if len(self.train_losses) > self.batch:
            train_l = self.train_losses[-self.batch:]
            test_l = self.test_losses[-self.batch:]

            train_diffs = np.array([train_l[i + 1] - train_l[i] for i in range(0, self.batch - 1)])
            test_diffs = np.array([test_l[i + 1] - test_l[i] for i in range(0, self.batch - 1)])

            avg_train_diff = np.average(train_diffs)
            avg_test_diff = np.average(test_diffs)

            print(f'Converging iter: {self.IsConverging}')
            print(f'Average diffs: {avg_train_diff}, {avg_test_diff}')
            if avg_test_diff >= self.MinDiff:
                self.IsConverging += 1
                self.NotConverging = 0
            else:
                self.NotConverging += 1

            if self.NotConverging == 1:
                self.IsConverging = 0

            if test_loss < self.best_test_loss:
                self.best_test_loss = test_loss
                # self.IsConverging = 0

            if self.IsConverging >= self.ConvergingMaxIter:
                return True
            if self.epoch > self.MaxIter and self.MaxIter != -1:
                return True
            else:
                return False

    def PlotLoss(self):
        global seed
        x = range(len(self.train_losses))
        plt.plot(x, self.train_losses, label='Train losses')
        plt.plot(x, self.test_losses, label='Test losses')

        x = [self.train_losses.index(self.best_train_loss), self.test_losses.index(self.best_test_loss)]
        y = [self.best_train_loss, self.best_test_loss]

        plt.scatter(x, y, color='red', s=100)
        plt.title(self.PlotTitle)
        plt.legend()
        plt.grid(True)
        plt.show()

    def save(self, filename):
        with open(filename, 'wb') as file:
            pickle.dump(self.__dict__, file)
        print(f"Parameters saved to {filename}")

    def load(self, filename):
        with open(filename, 'rb') as file:
            data = pickle.load(file)
            self.__dict__.update(data)
        print(f"Parameters loaded from {filename}")
