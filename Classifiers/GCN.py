import sys

import dgl
import dgl.data
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn.pytorch import GraphConv, AvgPooling, SumPooling
import networkx as nx
import matplotlib.pyplot as plt
import json
import numpy as np
import time
import pandas as pd
import random

last_percent = 0
t = 0
def EstimateTimePercent(i, maxiter, chunk):
    global t
    global last_percent
    if i % chunk == 0:
        percent = (i / maxiter) * 100

        if i != 0:
            diff = percent - last_percent
            interval = time.time() - t
            estimate_time = int((100 - percent) * interval / diff)
        else:
            estimate_time = -1

        t = time.time()
        last_percent = percent

        print(f"{percent:.3f}% - estimate time: {estimate_time}")

#PATHS
if 1:
    graph_path = r'C:\Users\xamuc\Desktop\PIC1\DataSetup\STRING\gene_links_graph.dgl'
    data_path = r'C:\Users\xamuc\Desktop\PIC1\DataSetup\GTex\gene_tpm_whole_blood_graph.csv'
    ground_truth_path = r'C:\Users\xamuc\Desktop\PIC1\DataSetup\GTex\gene_tpm_brain_frontal_cortex_graph.csv'
    model_path = r'C:\Users\xamuc\Desktop\PIC1\DataSetup\Model\fronta_cortex_predictor.pth'
    model2_path = r'C:\Users\xamuc\Desktop\PIC1\DataSetup\Model\fronta_cortex_predictor2.pth'

class GCN(nn.Module):
    def __init__(self, num_classes):
        super(GCN, self).__init__()

        h1 = 60
        h2 = 30
        self.conv = GraphConv(1, h1)
        self.pool = AvgPooling()

        self.lin1 = nn.Linear(h1, h2)
        self.lin2 = nn.Linear(h2, num_classes)

    def forward(self, g, in_feat):

        h = self.conv(g, in_feat)
        h = F.relu(h)
        h = self.pool(g, h)
        h = F.relu(h)
        h = self.lin1(h)
        h = F.relu(h)
        h = self.lin2(h)
        h = F.relu(h)
        return h

class EarlyStopper:
    def __init__(self):
        self.train_losses = []
        self.test_losses = []
        self.best_test_loss = 100
        self.IsConverging = 0
        self.NotConverging = 0
        self.ConvergingMaxIter = 5
        self.MinDiff = -10**-4
        self.batch = 4
    def Loss(self, train_loss, test_loss):
        self.train_losses += [train_loss.cpu().detach().numpy()]
        self.test_losses += [test_loss.cpu().detach().numpy()]

        if len(self.train_losses) > self.batch:
            train_l = self.train_losses[-self.batch:]
            test_l = self.test_losses[-self.batch:]

            train_diffs = np.array([train_l[i+1]-train_l[i] for i in range(0,self.batch-1)])
            test_diffs = np.array([test_l[i+1] - test_l[i] for i in range(0, self.batch-1)])

            avg_train_diff = np.average(train_diffs)
            avg_test_diff = np.average(test_diffs)

            print(f'Is converging: {self.IsConverging}')
            print(f'Average diffs: {avg_train_diff}, {avg_test_diff}')
            if avg_train_diff >= self.MinDiff or avg_test_diff >= self.MinDiff:
                self.IsConverging += 1
                self.NotConverging = 0
            else:
                self.NotConverging += 1

            if self.NotConverging == 1:
                self.IsConverging = 0


            if test_loss < self.best_test_loss:
                self.best_test_loss = test_loss
                #self.IsConverging = 0

            if self.IsConverging >= self.ConvergingMaxIter:
                return True
            else: return False

    def PlotLoss(self):
        x = range(len(self.train_losses))
        plt.plot(x, self.train_losses, label='Train losses')
        plt.plot(x, self.test_losses, label='Test losses')
        plt.title('Losses per epoch')
        plt.legend()
        plt.show()

def train(model, g, data, gt):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    best_test_loss = 0

    N = len(data)
    random_ints = random.sample(range(0, N), N)

    for e in range(100):
        # TRAINING --------------------------------------
        if 1:
            train_indexes = random_ints[:int(N * 0.7)]
            train_set = data[train_indexes]
            gt_train_set = gt[train_indexes]

            train_size = len(train_indexes)
            bs = 1
            bn = train_size // bs + 1
            for bi in range(bn):
                I = range(bi * bs, min((bi + 1) * bs, train_size))
                train_batch = train_set[I].transpose(0,1)
                gt_train_batch = gt_train_set[I]
                if list(I) != []:
                    pred = model(g, train_batch)
                    loss = F.mse_loss(pred, gt_train_batch)

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

        # TESTING
        if 1:
            test_indexes = random_ints[int(N * 0.7):]
            test_set = data[test_indexes]
            gt_test_set = gt[test_indexes]
            test_size = len(test_indexes)
            bs = 1
            bn = test_size // bs + 1
            for bi in range(bn):
                I = range(bi * bs, min((bi + 1) * bs, test_size))
                test_batch = test_set[I].transpose(0, 1)
                gt_test_batch = gt_test_set[I]
                if list(I) != []:
                    pred = model(g, test_batch)
                    test_loss = F.mse_loss(pred, gt_test_batch)

        if e % 1 == 0: print(f"In epoch {e}, train loss: {loss:.3f}, test loss: {test_loss:.3f}")
        EstimateTimePercent(e, 100, 1)
        EarlyStop = ES.Loss(loss, test_loss)
        if EarlyStop: break

#Define seed
if 1:
    seed = random.randint(0, (2**32)-1)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

#Load data
if 1:
    print("Loading graph...")
    g = dgl.load_graphs(graph_path)[0][0].to('cuda')
    print("Loading Data...")
    ddf = pd.DataFrame(pd.read_csv(data_path))
    gtdf = pd.DataFrame(pd.read_csv(ground_truth_path))
    ddf.iloc[:, 1:] = ddf.iloc[:, 1:].astype(float)
    gtdf.iloc[:, 1:] = gtdf.iloc[:, 1:].astype(float)

    data = torch.stack([torch.tensor(ddf.iloc[:, i], dtype=torch.float32) for i in range(1, len(ddf.columns))]).to('cuda')  # (159, 18678)
    gt = torch.stack([torch.tensor(gtdf.iloc[:, i], dtype=torch.float32) for i in range(1, len(gtdf.columns))]).to('cuda')  # (159, 15)
    print(f'data shape: {data.shape}')
    print(f'ground truth shape: {gt.shape}')

#Training code
if 1:
    print("Training...")
    best_train_loss = 100
    best_test_loss = 100
    n = 0
    while True:
        ES = EarlyStopper()
        model = GCN(gt.shape[1]).to('cuda')
        train(model, g, data, gt)
        print(f"Final training loss {ES.train_losses[-1]} and test loss {ES.test_losses[-1]}")

        if ES.test_losses[-1] < best_test_loss:
            best_test_loss = ES.test_losses[-1]
            torch.save(model, rf'C:\Users\xamuc\Desktop\PIC1\DataSetup\Model\fronta_cortex_predictor{n}.pth')
            print(f"Saved model {n}, seed: {seed}\n\n")
            n += 1
        elif ES.train_losses[-1]  < best_train_loss:
            best_train_loss = ES.train_losses[-1]
            torch.save(model, rf'C:\Users\xamuc\Desktop\PIC1\DataSetup\Model\fronta_cortex_predictor{n}.pth')
            print(f"Saved model {n}, seed: {seed}\n\n")
            n += 1

#Prediction code
if 0:
    print("Loading model...")
    model = torch.load(model_path).to('cuda')
    with torch.no_grad():
        for i in range(len(data)):
            model.eval()
            x = data[i].unsqueeze(1)
            pred = model(g, x)
            gti = gt[i]
            print(f'pred: {pred.tolist()[0]}')
            print(f'ground truth: {gti.tolist()}\n')