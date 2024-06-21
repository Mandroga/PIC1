import random

from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import mean_squared_error
import pandas as pd
import torch
import matplotlib.pyplot as plt
import numpy as np
import time
import joblib
from DLTools import *

project_dir = r'C:\Users\xamuc\Desktop\PIC1\Data'
#PATHS
if 1:
    data_path = project_dir + r'\GTEx\GTEx_wbg_df.csv'
    ground_truth_path = project_dir + r'\GTEx\GTEx_fcdf_filtered.csv'
    model_path = project_dir + r'\Predictors\MLP_amygdala_predictor.pkl'
    ES_path = project_dir + r'\Predictors\MLP_ES.pkl'

#Load data
if 1:
    ddf = pd.DataFrame(pd.read_csv(data_path))
    gtdf = pd.DataFrame(pd.read_csv(ground_truth_path))
#    ddf.iloc[:, 1:] = ddf.iloc[:, 1:].astype(float)
   # gtdf.iloc[:, 1:] = gtdf.iloc[:, 1:].astype(float)

    data = torch.stack([torch.tensor(ddf.iloc[:, i], dtype=torch.float32) for i in range(3, len(ddf.columns))])
    gt = torch.stack([torch.tensor(gtdf.iloc[:, i], dtype=torch.float32) for i in range(3, len(gtdf.columns))])

    X = data
    y = gt

#Train MLP
if 1:
    seed = SetSeed(1596973221)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed)

    # Initialize MLPRegressor
    mlp = MLPRegressor(hidden_layer_sizes=(5000, 1000, 100), activation='relu', solver='adam',
                       alpha=0.0001, batch_size='auto', learning_rate='constant',
                       learning_rate_init=0.001, max_iter=200, random_state=seed)

    ES = EarlyStopper()
    ETP = EstimateTimePercent()
    # Training loop
    print(f"training {seed}...")
    for epoch in range(mlp.max_iter):
        # Train the model for one epoch
        mlp.partial_fit(X_train, y_train)

        # Calculate train loss
        train_pred = mlp.predict(X_train)

        # Calculate test loss
        test_pred = mlp.predict(X_test)

        train_loss = mean_squared_error(y_train, train_pred)
        test_loss = mean_squared_error(y_test, test_pred)
        EarlyStop = ES.Loss(train_loss, test_loss)

        print(f'Epoch {epoch}| train loss: {train_loss}, test loss: {test_loss}')
        ETP.f(epoch+1, mlp.max_iter, 1)
        if EarlyStop: break

    ES.save(ES_path)
    joblib.dump(mlp, model_path)

#Load MLP
if 1:
    seed = SetSeed(1596973221)
    mlp = joblib.load(model_path)
    ES = EarlyStopper()
    ES.load(ES_path)

if 1:
    ES.PlotLoss()
    plt.title(f'MLP Loss per epoch')
    plt.show()
#Evaluate
if 1:
    pred_matrix = torch.tensor(mlp.predict(X))
    mean_matrix = torch.stack([torch.mean(gt, dim=0)] * gt.shape[0])
    E = Evaluate(pred_matrix, gt)
    E.std_measure()
    E.AvgResults()
    E = Evaluate(mean_matrix, gt)
    E.std_measure()
    E.AvgResults()

