import dgl.data
from DLTools import *
from GCNModels import *
from sklearn.model_selection import train_test_split
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset

def savefile(file, file_path):
    with open(file_path, 'wb') as f:
        return pickle.dump(file, f)
def loadfile(file_path):
    with open(file_path, 'rb') as f:
        return pickle.load(f)
def train(model, g, data, gt):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    train_set, test_set, gt_train_set, gt_test_set = train_test_split(data, gt, test_size=0.3, random_state=seed)
    ETP = EstimateTimePercent()
    e = -1
    while True:
        e += 1
        if 1:
            for i in range(len(train_set)):
                pred = model(g, train_set[i].unsqueeze(1))
                pred = pred[gt_index_in_geneset].T
                loss = F.mse_loss(pred, gt_train_set[i].unsqueeze(0))

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        # TESTING
        if 1:
            for i in range(len(test_set)):
                pred = model(g, test_set[i].unsqueeze(1))
                pred = pred[gt_index_in_geneset].T
                #print(f"test pred: {pred}")
                test_loss = F.mse_loss(pred, gt_test_set[i].unsqueeze(0))

        if e % 1 == 0: print(f"In epoch {e}, train loss: {loss:.3f}, test loss: {test_loss:.3f}")
        ETP.f(e, 100, 1)
        EarlyStop = ES.Loss(loss.item(), test_loss.item())
        if EarlyStop: break
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
            batch_gt = batch_gt
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
        chunk = 5
        if e % chunk == 0:
            print(f"In epoch {e}, train loss: {loss:.1e}, test loss: {test_loss:.1e}")
        ETP.f(e, ES.MaxIter, chunk)
        EarlyStop = ES.Loss(train_loss, test_loss)
        if EarlyStop: break

project_dir = r'C:\Users\xamuc\Desktop\PIC1\Data'
#PATHS
if 1:
    input_tissue = 'Whole blood'
    target_tissue = 'Amygdala'
    graph_path = project_dir + rf'\STRING\{input_tissue} - {target_tissue}\dgl_graph.bin'
    data_path = project_dir + rf'\GTEx\{input_tissue} - {target_tissue}\GTEx_input_graph_df.csv'
    ground_truth_path = project_dir + rf'\GTEx\{input_tissue} - {target_tissue}\GTEx_target_df_filtered.csv'

#Load data
if 1:
    print("Loading graph...")
    g = dgl.load_graphs(graph_path)[0][0].to('cuda')
    print("Loading Data...")
    ddf = pd.DataFrame(pd.read_csv(data_path))
    gtdf = pd.DataFrame(pd.read_csv(ground_truth_path))
    #ddf.iloc[:, 3:] = ddf.iloc[:, 3:].astype(float)
    #gtdf.iloc[:, 3:] = gtdf.iloc[:, 3:].astype(float)

    data = torch.stack([torch.tensor(ddf.iloc[:, i], dtype=torch.float32) for i in range(3, len(ddf.columns))]).to('cuda')  # (159, 18678)
    gt = torch.stack([torch.tensor(gtdf.iloc[:, i], dtype=torch.float32) for i in range(3, len(gtdf.columns))]).to('cuda')  # (159, 15)
    print(f'data shape: {data.shape}')
    print(f'ground truth shape: {gt.shape}')

    gt_index_in_geneset = [1917,8197,15122]

#Random search Training code
if 0:
    print("Training...")
    best_train_loss = 100
    best_test_loss = 100
    n = 0
    while True:
        seed = SetSeed()
        print(f'seed : {seed}')
        ES = EarlyStopper()
        ES.ConvergingMaxIter = 10
        model = GCN0(gt.shape[1]).to('cuda')
        train(model, g, data, gt)
        print(f"Final training loss {ES.train_losses[-1]}, test loss {ES.test_losses[-1]}, seed: {seed}\n\n")
        ES.PlotLoss()

        if ES.test_losses[-1] < best_test_loss:
            best_test_loss = ES.test_losses[-1]
            torch.save(model, rf'C:\Users\xamuc\Desktop\PIC1\DataSetup\Model\frontal_cortex_predictor{n}.pth')
            print(f"Saved model {n}")
            n += 1

#Train seed
if 1:
    seed = SetSeed(1596973221)
    ES = EarlyStopper()
    ES.ConvergingMaxIter = 50
    ES.MaxIter = 700
    model = GCN1(gt.shape[1]).to('cuda')
    optimizer = torch.optim.Adam(model.parameters(), lr=0.075)
    train_GCN_batch(model, optimizer, ES, g, data, gt)
    ES.PlotLoss()
    plt.title = f'GCN1 {seed} - Loss per epoch'
    plt.show()
    savefile(ES, project_dir + r'\Predictors\Earlystop.pkl')
    torch.save(model, project_dir + r'\Predictors\amygdala_predictor.pth')

#load
if 1:
    print("Loading model...")
    model = torch.load(project_dir + r'\Predictors\amygdala_predictor.pth').to('cuda')
    ES = loadfile(project_dir + r'\Predictors\Earlystop.pkl')

def PredMatrix(model, data):
    data = data.T.unsqueeze(2)
    model.eval()
    with torch.no_grad():
        pred = model(g, data).squeeze(0)
    print(pred.shape)
    return pred

#Evaluate
if 1:

    ES.PlotLoss()
    plt.title("GCN1 Loss per epoch")
    plt.show()

    pred_matrix = PredMatrix(model, data)
    mean_matrix = torch.stack([torch.mean(gt, dim=0)] * gt.shape[0]).to('cuda')

    E = Evaluate(pred_matrix, gt)
    E.measure(1)
    E.AvgResults()
    E = Evaluate(mean_matrix, gt)
    E.measure(1)
    E.AvgResults()

