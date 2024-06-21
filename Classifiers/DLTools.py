import numpy as np
import matplotlib.pyplot as plt
import pickle

import time
import random
import torch

class EarlyStopper:
    def __init__(self):
        self.train_losses = []
        self.test_losses = []

        self.best_train_loss = -1
        self.best_test_loss = -1

        self.IsConverging = 0
        self.NotConverging = 0
        self.ConvergingMaxIter = 5
        self.ClearConvergingIter = 1

        self.MinDiff = -10 ** -3
        self.BDfrac = 100
        self.batch = 2

        self.epoch = -1
        self.MaxIter = -1

        self.debug = True
        #methods
        self.BDF = True

        #Plot
        self.PlotTitle = 'Loss Per Epoch'

    def Loss(self, train_loss, test_loss):
        self.epoch += 1
        self.train_losses += [train_loss]
        self.test_losses += [test_loss]

        if test_loss < self.best_test_loss or self.best_test_loss == -1:
            self.best_test_loss = test_loss
        if train_loss < self.best_train_loss or self.best_train_loss == -1:
            self.best_train_loss = train_loss

        return self.Methods()
    #------------------------
    def Methods(self):
        if self.BDF: self.BatchDiffFrac()
        if self.MaxIter != -1: self.MaxIteration()
        return self.Converging()

    def MaxIteration(self):
        if self.epoch >= self.MaxIter:
            self.IsConverging = self.ConvergingMaxIter

    def BatchDiffFrac(self):
        #Gets a batch of test losses, calculates difference between epoch, averages differences
        # and if its higher than BDfrac of the current test loss assumes converging

        if len(self.train_losses) > self.batch:
            train_loss_batch = self.train_losses[-self.batch:]
            test_loss_batch = self.test_losses[-self.batch:]

            train_diffs = np.array([train_loss_batch[i + 1] - train_loss_batch[i] for i in range(0, self.batch - 1)])
            test_diffs = np.array([test_loss_batch[i + 1] - test_loss_batch[i] for i in range(0, self.batch - 1)])

            self.avg_train_diff = np.average(train_diffs)
            self.avg_test_diff = np.average(test_diffs)



            if self.avg_test_diff >= -abs(self.test_losses[-1]/self.BDfrac):
                self.IsConverging += 1
                self.NotConverging = 0
            else:
                self.NotConverging += 1
            if self.debug: self.BDFinfo()
    def BDFinfo(self):
        if len(self.train_losses) > self.batch and self.BDF:
            print(f'Average diffs: {self.avg_train_diff}, {self.avg_test_diff}')
            print(f'Min diff {-abs(self.test_losses[-1]/self.BDfrac)}')
            print(f'Converging iter: {self.IsConverging}')

    def Converging(self):
        if self.IsConverging == self.ConvergingMaxIter:
            return True
        if self.NotConverging == self.ClearConvergingIter:
            self.IsConverging = 0
        return False
    # ------------------------
    def PlotLoss(self):
        x = range(len(self.train_losses))
        plt.plot(x, self.train_losses, label='Train losses', marker='s', markersize=4)
        plt.plot(x, self.test_losses, label='Test losses', marker='s', markersize=4)

        x = [self.train_losses.index(self.best_train_loss), self.test_losses.index(self.best_test_loss)]
        y = [self.best_train_loss, self.best_test_loss]

        plt.scatter(x, y, color='red', s=100)
        #plt.title(self.PlotTitle)
        plt.legend()
        plt.yscale('log')
        plt.grid(True)
        #plt.show()

    def save(self, filename):
        with open(filename, 'wb') as file:
            pickle.dump(self.__dict__, file)
        print(f"Parameters saved to {filename}")
    def load(self, filename):
        with open(filename, 'rb') as file:
            data = pickle.load(file)
            self.__dict__.update(data)
        print(f"Parameters loaded from {filename}")

class Evaluate():
    def __init__(self, pred_matrix, gt_matrix):
        #Matrix input shape (#samples, #features)
        self.pred_matrix = pred_matrix.cpu()
        self.gt_matrix = gt_matrix.cpu()
        self.std = torch.std(self.gt_matrix, dim=0)
        self.results = -1
        self.average_results = -1

    def std_measure(self):
        self.results = torch.abs( (self.pred_matrix - self.gt_matrix) / self.std)
        self.average_results = torch.mean(self.results, dim=0)
    def measure(self,c):
        measure = torch.full(self.std.shape, c)
        self.results = torch.abs((self.pred_matrix - self.gt_matrix) / measure)
        self.average_results = torch.mean(self.results, dim=0)
    def relative_measure(self):
        self.results = torch.abs( (self.pred_matrix - self.gt_matrix) / self.gt_matrix)
        self.average_results = torch.mean(self.results, dim=0)

    def Results(self):
        torch.set_printoptions(threshold=float('inf'))
        print(self.results)
    def AvgResults(self):
        print(self.average_results.tolist())

class EstimateTimePercent():
    def __init__(self):
        self.t = 0
        self.last_percent = 0
    def f(self, i, maxiter, chunk):
        if i % chunk == 0:
            percent = (i / maxiter) * 100

            if i != 0:
                diff = percent - self.last_percent
                interval = time.time() - self.t
                estimate_time = int((100 - percent) * interval / diff)
            else:
                estimate_time = -1

            self.t = time.time()
            self.last_percent = percent

            print(f"{percent:.3f}% - estimate time: {estimate_time}")

def SetSeed(seed_ = -1):
    if seed_ == -1: seed = random.randint(0, (2**32)-1)
    else: seed = seed_
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    return seed