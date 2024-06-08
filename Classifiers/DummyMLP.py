from DLTools import *
import random
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import pandas as pd
import torch
import joblib

#PATH
if 1:
    graph_path = r'C:\Users\xamuc\Desktop\PIC1\DataSetup\Dummy\dummygraph.dgl'
    data_path = r'C:\Users\xamuc\Desktop\PIC1\DataSetup\Dummy\data.csv'
    model_path = r'C:\Users\xamuc\Desktop\PIC1\DataSetup\Dummy\dummyMLP.pkl'
    ES_path = r'C:\Users\xamuc\Desktop\PIC1\DataSetup\Model\DummyMLP_ES.pkl'

#Load data
if 1:
    print("Loading Data...")
    data = []
    gt = []
    for i in range(5):
        ddf = pd.DataFrame(pd.read_csv(rf'C:\Users\xamuc\Desktop\PIC1\DataSetup\Dummy\data{i + 1}.csv'))
        ddf = ddf.astype(float)

        data += [torch.stack([torch.tensor(ddf.iloc[i, :-1], dtype=torch.float32) for i in range(len(ddf))])]
        gt += [torch.stack([torch.tensor(ddf.iloc[i, -1], dtype=torch.float32) for i in range(len(ddf))]).unsqueeze(1)]
    X = data
    y = gt

#Train MLP
if 1:
    seed = SetSeed(1596973221)
    best_test_loss = []
    hs = [[36, 18, 6], [100, 50, 36], [200, 100, 50],[300,150,72]]
    #lrs = [0.001, 0.01, 0.1, 1]
    lrs = [0.1]
    for i in range(4,5):
        X_train, X_test, y_train, y_test = train_test_split(X[i], y[i], test_size=0.3, random_state=seed)
        y_train = y_train.squeeze(1)
        y_test = y_test.squeeze(1)
        for h in hs:
            for lr in lrs:
                # Initialize MLPRegressor
                h1,h2,h3 = h
                mlp = MLPRegressor(hidden_layer_sizes=(h1, h2, h3), activation='relu', solver='adam',
                                   alpha=0.0001, batch_size='auto', learning_rate='constant',
                                   learning_rate_init=lr, random_state=seed)

                ES = EarlyStopper()
                ES.BDF = False
                ES.MaxIter = 600
                ETP = EstimateTimePercent()
                # Training loop
                print(f'Training {i}, {h}, lr = {lr}')
                epoch = -1
                while True:
                    epoch += 1
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
                        ETP.f(epoch + 1, mlp.max_iter, 50)
                    if EarlyStop: break

                best_test_loss += [ES.best_test_loss]

                ES.PlotLoss()
                plt.title(f'MLP {h}_{lr:.3f}')
                plt.legend()
                plt.grid(True)
                plt.savefig(
                    rf'C:\Users\xamuc\Desktop\PIC1\DataSetup\Model\Dummy\DataSet{i + 1}\MLP\MLP-{h}_{lr:.3f}.png')
                #plt.show()
                plt.clf()

                print(f'Best test losses\nMLP: {best_test_loss}')

def train(mlp, ES, Xi, yi):
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
def GridTrain(model_name, hs, lrs, data_index_list):
    seed = SetSeed(1596973221)
    for i in data_index_list:
        best_test_loss = []
        for h in hs:
            for lr in lrs:
                mlp = ChooseModel(model_name, h, lr)

                ES = EarlyStopper()
                ES.BDF = False
                ES.MaxIter = 600

                print(f'Training DataSet{i+1}, {h}_{lr}')
                train(mlp, ES, X[i], y[i])
                best_test_loss += [ES.best_test_loss]

                ES.PlotLoss()
                title = f'MLP {h}_{lr:.3f}'
                plt.title(title)
                plt.legend()
                plt.grid(True)
                plt.savefig(rf'C:\Users\xamuc\Desktop\PIC1\DataSetup\Model\Dummy\DataSet{i + 1}\MLP\MLP-{h}_{lr:.3f}.png')
                #plt.show()
                plt.clf()

                print(f'Best test losses\nMLP: {best_test_loss}')

def ChooseModel(model_name, h, lr):
    if model_name == '3layer':
        h1, h2, h3 = h
        return MLPRegressor(hidden_layer_sizes=(h1, h2, h3), activation='relu', solver='adam',
                                   alpha=0.0001, batch_size='auto', learning_rate='constant',
                                   learning_rate_init=lr, random_state=seed)

#Load MLP
if 0:
    seed = SetSeed(97733834)
    mlp = joblib.load(model_path)
    ES = EarlyStopper()
    ES.load(ES_path)

#Evaluate
if 0:
    for i in range(3):
        pred_matrix = torch.tensor(mlp.predict(X[i]))
        mean_matrix = torch.stack([torch.mean(y[i], dim=0)] * 100)
        E = Evaluate(pred_matrix, y[i])
        E.std_measure()
        #E.Results()
        E.AvgResults()
    #E = Evaluate(mean_matrix, gt)
    #E.std_measure()
   # E.AvgResults()



ES.PlotTitle = f'Dummy MLP {seed} - Loss per epoch'
ES.PlotLoss()