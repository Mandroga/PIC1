import os
import json

from DLTools import *
from GCNModels import *

from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

import dgl.data
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

import pandas as pd

def GenerateDir(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)

def ManageBestLoss(best_test_loss_dict, file_name='best_test_loss'):
    with open(project_dir + rf'\Dummy\Losses\{file_name}.json', 'w') as f:
        json.dump(best_test_loss_dict, f)
    with open(project_dir + rf'\Dummy\Losses\{file_name}.txt', 'w') as f:
        for DataSet in list(best_test_loss_dict['Data']):
            f.write(DataSet+'\n')
            for Model in list(best_test_loss_dict['Data'][DataSet]):
                f.write(f"{Model}:{best_test_loss_dict['Data'][DataSet][Model]}\n")
        f.write('\n')
        for Model in list(best_test_loss_dict['Parameters']):
            f.write(f"{Model}:{best_test_loss_dict['Parameters'][Model]}\n")




def train_GCN_batch(model, optimizer, ES, g, data, gt, batch_size=32):
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

def train_MLP(mlp, ES, Xi, yi):
    X_train, X_test, y_train, y_test = train_test_split(Xi, yi, test_size=0.3, random_state=seed)
    y_train = y_train.squeeze(1)
    y_test = y_test.squeeze(1)
    epoch = 0
    ETP = EstimateTimePercent()
    while True:
        # Train the model for one epoch
        mlp.partial_fit(X_train, y_train)
        # Calculate train loss
        train_pred = mlp.predict(X_train)
        # Calculate test loss
        test_pred = mlp.predict(X_test)

        train_loss = mean_squared_error(y_train, train_pred)
        test_loss = mean_squared_error(y_test, test_pred)
        EarlyStop = ES.Loss(train_loss, test_loss)

        if epoch % 50 == 0:
            print(f'Epoch {epoch} | train loss: {train_loss}, test loss: {test_loss}')
            ES.BDFinfo()
            ETP.f(epoch + 1, ES.MaxIter, 50)
        if EarlyStop: break
        epoch += 1

def ChooseModelTrain(model_name, h, lr, g, data_i, gt_i, ES):
    if model_name == 'GCN1':
        h1, h2, h3 = h
        model = DummyGCN1(h1, h2, h3).to('cuda')
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        train_GCN_batch(model, optimizer, ES, g, data_i, gt_i)

    if model_name == 'GCN2':
        h1, h2, h3 = h
        model = DummyGCN2(h1, h2, h3).to('cuda')
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        train_GCN_batch(model, optimizer, ES, g, data_i, gt_i)

    if model_name == 'GCN3':
        model = DummyGCN3(h).to('cuda')
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        train_GCN_batch(model, optimizer, ES, g, data_i, gt_i)

    if model_name == 'MLP':
        model = MLPRegressor(hidden_layer_sizes=h, activation='relu', solver='adam',
                                   alpha=0.0001, batch_size='auto', learning_rate='constant',
                                   learning_rate_init=lr, random_state=seed)
        train_MLP(model, ES, data_i.cpu(), gt_i.cpu())
    return model

def GridListTrain(model_name, hs, lrs, data_packs, best_test_loss_dict):
    print(f'Training {model_name}')
    for i, data_i, gt_i , g in data_packs:
        best_test_loss = []
        for h in hs:
            for lr in lrs:
                print(f'Training DataSet{i}, {h}_{lr}')

                ES = EarlyStopper()
                ES.BDF = False
                ES.debug = False
                ES.MaxIter = 600

                ChooseModelTrain(model_name, h, lr, g, data_i, gt_i, ES)
                best_test_loss += [ES.best_test_loss]

                ES.PlotLoss()
                title = f'Dummy{model_name} {h}_{lr:.3f}'
                plt.title(title)
                plt.legend()
                plt.grid(True)

                save_dir = project_dir + rf'\Dummy\Losses\DataSet{i}\{model_name}'
                GenerateDir(save_dir)
                plt.savefig(save_dir + rf'\{title}.png')
                plt.clf()

        if 'Data' not in best_test_loss_dict:
            best_test_loss_dict['Data'] = {}
        if f'DataSet{i}' not in best_test_loss_dict['Data']:
            best_test_loss_dict['Data'][f'DataSet{i}'] = {}
        best_test_loss_dict['Data'][f'DataSet{i}'][model_name] = best_test_loss

        print(f'best_test_loss DataSet{i} - \n{model_name}:{best_test_loss}')

    if f'Parameters' not in best_test_loss_dict:
        best_test_loss_dict['Parameters'] = {}
    best_test_loss_dict['Parameters'][model_name] = {'hs':hs, 'lrs':lrs}

project_dir = rf'C:\Users\xamuc\Desktop\PIC1'

#Load data
if 0:
    print("Loading graphs...")
    gs = []
    for i in range(3):
        gs += [dgl.load_graphs(project_dir + rf'\Dummy\Data\graph{i}.dgl')[0][0].to('cuda')]

    print("Loading Data...")
    data = []
    gt = []
    for i in range(5):
        ddf = pd.DataFrame(pd.read_csv(project_dir + rf'\Dummy\Data\data{i}.csv'))
        ddf = ddf.astype(float)

        data += [torch.stack([torch.tensor(ddf.iloc[i, :-1], dtype=torch.float32) for i in range(len(ddf))]).to('cuda')]
        gt += [torch.stack([torch.tensor(ddf.iloc[i, -1], dtype=torch.float32) for i in range(len(ddf))]).to('cuda').unsqueeze(1)]

    data_packs = []
    for i in range(2):
        data_packs += [(i, data[i],gt[i],gs[0])]
    data_packs += [(2, data[2], gt[2], gs[1])]
    for i in range(3,5):
        data_packs += [(i, data[i], gt[i], gs[2])]

#Train
if 0:
    seed = SetSeed(1596973221)
    best_test_loss_dict = {}
    hs = [(1, 6, 1), (1, 36, 1), (6, 36, 6), (36, 36, 36), (12, 72, 24), (6, 108, 72)]
    lrs = [0.001, 0.01, 0.1, 1]
    GridListTrain('GCN1', hs, lrs, data_packs, best_test_loss_dict)
    GridListTrain('GCN2', hs, lrs, data_packs, best_test_loss_dict)
    hs = [(36, 18, 6), (100, 50, 36), (200, 100, 50), (300, 150, 72)]
    GridListTrain('MLP', hs, lrs, data_packs, best_test_loss_dict)
    hs = [6,12,24,48]
    lrs = [0.1]
    GridListTrain('GCN3', hs, lrs, data_packs, best_test_loss_dict)

    #ManageBestLosses(best_test_loss_dict)

#Best Test Losses Analysis
if 1:
    with open(project_dir + rf'\Dummy\Losses\best_test_loss_dict.json', 'r') as f:
        best_test_loss_dict = json.load(f)

    #Sorted best losses
    for DataSet in list(best_test_loss_dict['Data']):
        print(DataSet)
        for Model in list(best_test_loss_dict['Data'][DataSet]):
            print(f"{Model}:{sorted(best_test_loss_dict['Data'][DataSet][Model])[0]:.3f}")
