import os
import json
import pickle

import matplotlib.pyplot as plt

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

def savefile(file, file_dir):
    with open(file_dir, 'wb') as f:
        pickle.dump(file, f)
def loadfile(file_dir):
    with open(file_dir, 'rb') as f:
        return pickle.load(f)

def ManageBestLoss(best_test_loss_dict, file_name='best_test_loss'):
    with open(project_dir + rf'\Dummy\Losses\{file_name}.json', 'w') as f:
        json.dump(best_test_loss_dict, f)
    with open(project_dir + rf'\Dummy\Losses\{file_name}.txt', 'w') as f:
        for DataSet in best_test_loss_dict:
            f.write(DataSet+'\n')
            for Model in best_test_loss_dict[DataSet]:
                model_losses = ''
                for hiperpar in best_test_loss_dict[DataSet][Model]:
                    model_losses += f'{hiperpar} '
                print(f"{Model}:{model_losses}")
                f.write(f"{Model}:{model_losses}\n")

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
            print(f"In epoch {e}, train loss: {loss:.1e}, test loss: {test_loss:.1e}")
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

    if model_name == 'GCN4':
        h1, h2, h3 = h
        model = DummyGCN4(h1, h2, h3).to('cuda')
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
    model_pack = []
    for i, data_i, gt_i , g in data_packs:
        best_test_loss = []
        for h in hs:
            for lr in lrs:
                print(f'Training DataSet{i}, {h}_{lr}')

                ES = EarlyStopper()
                ES.BDF = False
                ES.debug = False
                ES.MaxIter = 600

                model = ChooseModelTrain(model_name, h, lr, g, data_i, gt_i, ES)
                model_pack += [(model_name, h, lr, g, data_i, gt_i, model)]

                ES.PlotLoss()
                title = f'Dummy{model_name} {h}_{lr:.3f}'
                plt.title(title)
                plt.legend()
                plt.grid(True)

                save_dir = project_dir + rf'\Dummy\Losses\DataSet{i}\{model_name}'
                GenerateDir(save_dir)
                plt.savefig(save_dir + rf'\{title}.png')
                plt.clf()

                if f'DataSet{i}' not in best_test_loss_dict:
                    best_test_loss_dict[f'DataSet{i}'] = {}
                if model_name not in best_test_loss_dict[f'DataSet{i}']:
                    best_test_loss_dict[f'DataSet{i}'][model_name] = []
                best_test_loss_dict[f'DataSet{i}'][model_name] += [(ES.best_test_loss, h, lr)]

        print(f'best_test_loss DataSet{i} - \n{model_name}:{ES.best_test_loss}')
    return model_pack

project_dir = rf'C:\Users\xamuc\Desktop\PIC1'
data_dir = project_dir + rf'\Dummy\Data'

#Load data
if 1:
    print("Loading graphs...")
    gs = []
    for i in range(3):
        gs += [dgl.load_graphs(data_dir + rf'\graph{i}.dgl')[0][0].to('cuda')]

    print("Loading Data...")
    data = []
    gt = []
    for i in range(10):
        ddf = pd.DataFrame(pd.read_csv(data_dir + rf'\data{i}.csv'))
        ddf = ddf.astype(float)

        data += [torch.stack([torch.tensor(ddf.iloc[i, :-1], dtype=torch.float32) for i in range(len(ddf))]).to('cuda')]
        gt += [torch.stack([torch.tensor(ddf.iloc[i, -1], dtype=torch.float32) for i in range(len(ddf))]).to('cuda').unsqueeze(1)]

    data_packs = []
    for i in range(2):
        data_packs += [(i, data[i],gt[i],gs[0])]
    data_packs += [(2, data[2], gt[2], gs[1])]
    for i in range(3,5):
        data_packs += [(i, data[i], gt[i], gs[2])]

    for i in range(5,10):
        data_packs += [(i, data[i], gt[i], gs[0])]

seed = SetSeed(1596973221)

#Train
if 0:
    #Grid search
    if 0:
        with open(project_dir + rf'\Dummy\Losses\best_test_loss_dict.json', 'r') as f:
            best_test_loss_dict = json.load(f)
        #best_test_loss_dict = {}
        hs = [(1, 6, 1), (1, 36, 1), (6, 36, 6), (36, 36, 36), (12, 72, 24), (6, 108, 72)]
        lrs = [0.001, 0.01, 0.025, 0.05, 0.075, 0.1, 1]
        GridListTrain('GCN1', hs, lrs, data_packs, best_test_loss_dict)
        GridListTrain('GCN2', hs, lrs, data_packs, best_test_loss_dict)
        hs = [(36, 18, 6), (100, 50, 36), (200, 100, 50), (300, 150, 72)]
        GridListTrain('MLP', hs, lrs, data_packs, best_test_loss_dict)
        hs = [6,12,24,48]
        GridListTrain('GCN3', hs, lrs, data_packs, best_test_loss_dict)
        ManageBestLoss(best_test_loss_dict, 'best_test_loss_dict')

    #Model pack
    if 1:
        model_pack = []
        best_test_loss_dict = {}
        hs = [(300, 150, 72)]
        lrs = [0.1]
        model_pack += GridListTrain('MLP', hs, lrs, data_packs, best_test_loss_dict)
        hs = [(6, 36, 6)]
        lrs = [0.075]
        #model_pack += GridListTrain('GCN1', hs, lrs, data_packs[5:], best_test_loss_dict)
        #model_pack += GridListTrain('GCN2', hs, lrs, data_packs[5:], best_test_loss_dict)
        hs = [24]
        lrs = [0.001]
        #model_pack += GridListTrain('GCN3', hs, lrs, data_packs[5:], best_test_loss_dict)

        savefile(model_pack, data_dir + r'\model_pack3.pkl')
        ManageBestLoss(best_test_loss_dict, 'best_test_loss_dict3')

#Evaluate
if 0:
    print("Evaluate")
    model_pack = loadfile(data_dir + r'\model_pack2.pkl')
    #print([pack[0] for pack in model_pack])
    for i in range(5):
        print(f'Data set {i+5}')
        packs = [model_pack[j] for j in range(i, len(model_pack), 5)]
        #print([pack[0] for pack in packs])
        pack = packs[0]
        print(f'{pack[0]}: {pack[1]}_{pack[2]}')
        data = pack[4].cpu()
        gt = pack[5].cpu()
        X_train, X_test, y_train, y_test = train_test_split(data, gt, test_size=0.3, random_state=seed)
        model = pack[6]
        pred_matrix = torch.tensor(model.predict(X_test)).unsqueeze(1)
        EVAL = Evaluate(pred_matrix, y_test)
        EVAL.measure(1)
       # print("MAE")
        #EVAL.AvgResults()
        EVAL.relative_measure()
        print("Relative error")
        EVAL.AvgResults()
        #print("STD error")
        EVAL.std_measure()
        #EVAL.AvgResults()
    
        for pack_j in packs[1:]:
            g = pack_j[3]
            data = pack_j[4]
            gt = pack_j[5]
            X_train, X_test, y_train, y_test = train_test_split(data, gt, test_size=0.3, random_state=seed)
            model = pack_j[6]
            model.eval()
            with torch.no_grad():
                pred = model(g, X_test.T.unsqueeze(2))
            EVAL = Evaluate(pred, y_test)
            print(f'{pack_j[0]}: {pack_j[1]}_{pack_j[2]}')
            for val in gt:
                if val == 0:
                    print(val)
            EVAL.measure(1)
            #print("MAE")
            #EVAL.AvgResults()
            EVAL.relative_measure()
           # print("Relative error")
            #EVAL.AvgResults()
            #print("STD error")
            EVAL.std_measure()
           # EVAL.AvgResults()
        print()


#Best Test Losses Analysis
if 0:
    with open(project_dir + rf'\Dummy\Losses\best_test_loss_dict2.json', 'r') as f:
        best_test_loss_dict = json.load(f)
    models = ['MLP', 'GCN1', 'GCN2', 'GCN3']
    #Sorted best losses
    if 1:
        print("Sorted best losses")
        for DataSet in best_test_loss_dict:
            print(DataSet)
            for Model in best_test_loss_dict[DataSet]:
                print(f"{Model}:{sorted(best_test_loss_dict[DataSet][Model], key=lambda x: x[0])}")

    #Best layer and learning rate
    if 0:
        print("Best layer and learning rate")
        for Model in models:
            j = 0
            best_h = {}
            best_lr = {}
            print(Model)
            for DataSet in best_test_loss_dict:
                best_loss, h, lr = sorted(best_test_loss_dict[DataSet][Model], key=lambda x: x[0])[0]
                if best_loss < 100:
                    if str(h) not in best_h: best_h[str(h)] = 1
                    else: best_h[str(h)] += 1
                    if str(lr) not in best_lr: best_lr[str(lr)] = 1
                    else: best_lr[str(lr)] += 1
            print(best_h, best_lr)

    #Best loss per dataset table
    if 0:
        print("Best loss per dataset table")
        print(" Data set & MLP & GCN1 & GCN2 & GCN 3 \\\\ \n\hline")
        for DataSet in best_test_loss_dict:
            best_losses = []
            for Model in models:
                best_losses += [sorted(best_test_loss_dict[DataSet][Model], key=lambda x: x[0])[0][0]]
            print(f'{DataSet[-1] } & {best_losses[0]/best_losses[0]:.1e} & {best_losses[1]/best_losses[0]:.1e} & {best_losses[2]/best_losses[0]:.1e} & {best_losses[3]/best_losses[2]:.1e} \\\\ ')

    #Best loss per dataset table normalized
    if 1:
        print("Best loss per dataset table normalized")
        print(" Data set & MLP & GCN1 & GCN2 & GCN 3 \\\\ \n\hline")
        for DataSet in best_test_loss_dict:
            best_losses = []
            for Model in models:
               # print(Model)
                best_losses += [sorted(best_test_loss_dict[DataSet][Model], key=lambda x: x[0])[0][0]]
            print(f'{DataSet[-1] } & {best_losses[0]/best_losses[0]:.1e} & {best_losses[1]/best_losses[0]:.1e} & {best_losses[2]/best_losses[0]:.1e} & {best_losses[3]/best_losses[2]:.1e} \\\\ ')

    #plots
    if 0:
        colors = ['red', 'blue', 'green', 'purple', 'orange', 'black']
        lrs = [0.001, 0.01, 0.025, 0.05, 0.075, 0.1, 1]
        for model in models:
            j = 0
            for DataSet in best_test_loss_dict:
                x = []
                y = []

                for data in best_test_loss_dict[DataSet][model]:
                    best_loss = data[0]
                    lr = data[2]
                    x += [lr]
                    y += [best_loss]

                plt.scatter(x,y, color=colors[j], label=f'DataSet{j}')
                j += 1
            plt.title(f'{model}')
            plt.xlabel('Learning rates')
            plt.ylabel('Best loss')
            plt.xscale('log')
            plt.yscale('log')
            plt.grid(True)
            plt.legend()
            #plt.show()
