from DLTools import *
from GCNModels import *
import dgl.data
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd

from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error


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

    data_packs = []
    for i in range(3):
        data_packs += [(i, data[i],gt[i],gs[0])]
    for i in range(3,5):
        data_packs += [(i, data[i], gt[i], gs[1])]

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
        model = DummyGCN1(h).to('cuda')
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        train_GCN_batch(model, optimizer, ES, g, data_i, gt_i)

    if model_name == 'MLP':
        mlp = MLPRegressor(hidden_layer_sizes=h, activation='relu', solver='adam',
                                   alpha=0.0001, batch_size='auto', learning_rate='constant',
                                   learning_rate_init=lr, random_state=seed)
        train_MLP(mlp, ES, data_i, gt_i)

def GridListTrain(model_name, hs, lrs, data_packs):
    print(f'Training {model_name}')
    for i, data_i, gt_i , g in data_packs:
        best_test_loss = []
        for h in hs:
            for lr in lrs:
                print(f'Training DataSet{i+1}, {h}_{lr}')

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
                plt.savefig(rf'C:\Users\xamuc\Desktop\PIC1\DataSetup\Model\Dummy\DataSet{i + 1}\{model_name}\{title}.png')
                plt.clf()
        print(f'best_test_loss DataSet{i+1} - \n{model_name}:{best_test_loss}')

#Train
if 1:
    seed = SetSeed(97733834)
    hs = [(36, 18, 6), (100, 50, 36), (200, 100, 50), (300, 150, 72)]
    lrs = [0.1]
    GridListTrain('MLP', hs, lrs, data_packs)
