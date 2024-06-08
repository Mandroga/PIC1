from DLTools import *
from GCNModels import *
from sklearn.model_selection import train_test_split
import dgl.data
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import random

def train_stochastic(model, optimizer, g, data, gt):
    #optimizer = torch.optim.Adam(model.parameters(), lr=0.1)

    train_set, test_set, gt_train_set, gt_test_set = train_test_split(data, gt, test_size=0.3, random_state=seed)
    ETP = EstimateTimePercent()
    e = 0
    while True:
        # TRAINING --------------------------------------
        if 1:
            for i in range(len(train_set)):
                pred = model(g, train_set[i].unsqueeze(1))
                loss = F.mse_loss(pred, gt_train_set[i])
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        # TESTING
        if 1:
            for i in range(len(test_set)):
                pred = model(g, test_set[i].unsqueeze(1))
                test_loss = F.mse_loss(pred, gt_test_set[i])

        if e % 1 == 0: print(f"In epoch {e}, train loss: {loss:.3f}, test loss: {test_loss:.3f}")
        ETP.f(e, ES.MaxIter, 1)
        EarlyStop = ES.Loss(loss.item(), test_loss.item())
        if EarlyStop: break
        e += 1

def train_batch(model, optimizer, ES, g, data, gt, batch_size=32):
    train_set, test_set, gt_train_set, gt_test_set = train_test_split(data, gt, test_size=0.3, random_state=seed)

    # Create DataLoader for batching
    train_dataset = TensorDataset(train_set, gt_train_set)
    test_dataset = TensorDataset(test_set, gt_test_set)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    ETP = EstimateTimePercent()
    e = -1
    while True:
        e+=1
        model.train()
        train_loss = 0.0
        for batch_data, batch_gt in train_loader:
            batch_data = batch_data.T.unsqueeze(2)
            pred = model(g, batch_data)
            #print(pred.shape)
            #print(batch_gt.shape)
            loss = F.mse_loss(pred, batch_gt)
            train_loss += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        train_loss /= len(train_loader.dataset)
        # Evaluate on the test set
        model.eval()
        test_loss = 0.0
        with torch.no_grad():
            for batch_data, batch_gt in test_loader:
                batch_data = batch_data.T.unsqueeze(2)
                pred = model(g, batch_data)
                test_loss += F.mse_loss(pred, batch_gt, reduction='sum').item()

        test_loss /= len(test_loader.dataset)
        chunk = 50
        if e % chunk == 0:
            print(f"In epoch {e}, train loss: {loss:.3f}, test loss: {test_loss:.3f}")
        ETP.f(e, ES.MaxIter, chunk)
        EarlyStop = ES.Loss(train_loss, test_loss)
        if EarlyStop: break

def train(model, optimizer, g, data, gt):
    train_set, test_set, gt_train_set, gt_test_set = train_test_split(data, gt, test_size=0.3, random_state=seed)
    train_set = train_set.T.unsqueeze(2)
    test_set = test_set.T.unsqueeze(2)
    ETP = EstimateTimePercent()
    e = 0
    while True:
        # TRAINING --------------------------------------
        pred = model(g, train_set)
        loss = F.mse_loss(pred, gt_train_set)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # TESTING
        pred = model(g, test_set)
        test_loss = F.mse_loss(pred, gt_test_set)

        if e % 20 == 0: print(f"In epoch {e}, train loss: {loss:.3f}, test loss: {test_loss:.3f}")
        ETP.f(e, ES.MaxIter, 20)
        EarlyStop = ES.Loss(loss.item(), test_loss.item())
        if EarlyStop: break
        e += 1


#Load data
if 1:
    print("Loading graph...")
    gs = []
    for i in range(2):
        gs += [dgl.load_graphs(rf'C:\Users\xamuc\Desktop\PIC1\DataSetup\Dummy\dummygraph{i+1}.dgl')[0][0].to('cuda')]
    print("Loading Data...")
    data = []
    gt = []
    for i in range(5):
        ddf = pd.DataFrame(pd.read_csv(rf'C:\Users\xamuc\Desktop\PIC1\DataSetup\Dummy\data{i+1}.csv'))
        ddf = ddf.astype(float)

        data += [torch.stack([torch.tensor(ddf.iloc[i, :-1], dtype=torch.float32) for i in range(len(ddf))]).to('cuda')]
        gt += [torch.stack([torch.tensor(ddf.iloc[i, -1], dtype=torch.float32) for i in range(len(ddf))]).to('cuda').unsqueeze(1)]

    data_pack = []
    for i in range(3):
        data_pack += [(i, data[i],gt[i],gs[0])]
    for i in range(3,5):
        data_pack += [(i, data[i], gt[i], gs[1])]
    #print(len(data_pack[0]))
#Train
if 0:
    seed = SetSeed(1596973221)

    best_test_loss = []
    best_test_loss2 = []

    hs = [[1,6,1],[1,12,1],[1,24,1],[1,48,1]]
    # hs = [[1,6,1],[1,36,1],[6,36,6],[36,36,36],[12,72,24],[6,108,72]]
    #lrs = [0.001, 0.01, 0.1, 1]
    lrs = [0.1]
    for i in range(5):
        print(f'best_test_loss DataSet{i} - \nGCN1:{best_test_loss}\nGCN2:{best_test_loss2}')
        best_test_loss = []
        best_test_loss2 = []
        data_i = data[i]
        gt_i = gt[i]

        if i < 3:
            g = gs[0]
        elif i > 2 and i < 5:
            g = gs[1]

        for h in hs:
            for lr in lrs:
                h1, h2, h3 = h
                print(f'Training {i}, {h}, lr = {lr}')

                #DummyGCN1
                if 0:
                    ES = EarlyStopper()
                    ES.BDF = False
                    ES.debug = False
                    ES.MaxIter = 600

                    model = DummyGCN1(h1, h2, h3).to('cuda')
                    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
                    train_batch(model, optimizer, ES, g, data_i, gt_i)
                    best_test_loss += [ES.best_test_loss]

                    ES.PlotLoss()
                    plt.title(f'DummyGCN1 {h}_{lr:.3f}')
                    plt.legend()
                    plt.grid(True)
                    plt.savefig(
                        rf'C:\Users\xamuc\Desktop\PIC1\DataSetup\Model\Dummy\DataSet{i + 1}\GCN1\DummyGCN1-{h}_{lr:.3f}.png')
                    plt.clf()

                #Dummy GCN2
                if 0:
                    ES2 = EarlyStopper()
                    ES2.BDF = False
                    ES2.debug = False
                    ES2.MaxIter = 600

                    model2 = DummyGCN2(h1, h2, h3).to('cuda')
                    optimizer2 = torch.optim.Adam(model2.parameters(), lr=lr)
                    train_batch(model2, optimizer2, ES2, g, data_i, gt_i)
                    best_test_loss2 += [ES2.best_test_loss]

                    ES2.PlotLoss()
                    plt.title(f'DummyGCN2_batch {h}_{lr:.3f}')
                    plt.legend()
                    plt.grid(True)
                    plt.savefig(rf'C:\Users\xamuc\Desktop\PIC1\DataSetup\Model\Dummy\DataSet{i + 1}\GCN2\DummyGCN2_batch-{h}_{lr:.3f}.png')
                    plt.clf()

                # DummyGCN3
                if 1:
                    ES = EarlyStopper()
                    ES.BDF = False
                    ES.debug = False
                    ES.MaxIter = 600

                    model = DummyGCN3(h2).to('cuda')
                    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
                    train_batch(model, optimizer, ES, g, data_i, gt_i)
                    best_test_loss += [ES.best_test_loss]

                    ES.PlotLoss()
                    plt.title(f'DummyGCN3 {h2}_{lr:.3f}')
                    plt.legend()
                    plt.grid(True)
                    plt.savefig(
                        rf'C:\Users\xamuc\Desktop\PIC1\DataSetup\Model\Dummy\DataSet{i + 1}\GCN3\DummyGCN3-{h2}_{lr:.3f}.png')
                    plt.clf()
                #print(f'Best test losses\nGCN1:{best_test_loss}\nGCN2:{best_test_loss2}')

def ChooseModel(model_name, h):
    if model_name == 'GCN1':
        h1,h2,h3 = h
        return DummyGCN1(h1,h2,h3).to('cuda')
    if model_name == 'GCN2':
        h1,h2,h3 = h
        return DummyGCN2(h1,h2,h3).to('cuda')
    if model_name == 'GCN3':
        return DummyGCN3(h).to('cuda')
def GridTrain(model_name, hs, lrs, data_pack):
    for i, data_i, gt_i, g in data_pack:
        best_test_loss = []
        for h in hs:
            for lr in lrs:
                print(f'Training DataSet{i+1}, {h}_{lr}')

                ES = EarlyStopper()
                ES.BDF = False
                ES.debug = False
                ES.MaxIter = 600

                model = ChooseModel(model_name, h)
                optimizer = torch.optim.Adam(model.parameters(), lr=lr)
                train_batch(model, optimizer, ES, g, data_i, gt_i)
                best_test_loss += [ES.best_test_loss]

                ES.PlotLoss()
                title = f'Dummy{model_name} {h}_{lr:.3f}'
                plt.title(title)
                plt.legend()
                plt.grid(True)
                plt.savefig(rf'C:\Users\xamuc\Desktop\PIC1\DataSetup\Model\Dummy\DataSet{i + 1}\{model_name}\{title}.png')
                plt.clf()
        print(f'best_test_loss DataSet{i+1} - \n{model_name}:{best_test_loss}')

#Grid Train
if 1:
    seed = SetSeed(1596973221)
    hs = [[1, 6, 1], [1, 36, 1], [6, 36, 6], [36, 36, 36], [12, 72, 24], [6, 108, 72]]
    lrs = [0.001, 0.01, 0.1, 1]
    #GridTrain('GCN1', hs, lrs, data_pack)
    #GridTrain('GCN2', hs, lrs, data_pack)

    hs = [100]
    lrs = [0.1]
    GridTrain('GCN3', hs, lrs, [data_pack[2],])


def PredMatrix(model, data):
    model.eval()
    preds = []
    for i in range(len(data)):
        x = data[i].unsqueeze(1)
        with torch.no_grad():
            pred = model(g, x)
            pred = pred[3]
        preds.append(pred)
    preds = torch.stack(preds)
    return preds

#Evaluate
if 0:
    print("Loading model...")
    model = torch.load(r'C:\Users\xamuc\Desktop\PIC1\DataSetup\Model\frontal_cortex_predictor.pth').to('cuda')

    pred_matrix = PredMatrix(model, data)
    mean_matrix = torch.stack([torch.mean(gt, dim=0)] * 100).to('cuda')

    E = Evaluate(pred_matrix, gt)
    E.std_measure()
    E.Results()
    E.AvgResults()
    #E = Evaluate(mean_matrix, gt)
    #E.std_measure()
    #E.AvgResults()