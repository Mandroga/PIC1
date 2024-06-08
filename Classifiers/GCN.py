import dgl.data
from DLTools import *
from GCNModels import *
from sklearn.model_selection import train_test_split
import pandas as pd

#PATHS
if 1:
    graph_path = r'C:\Users\xamuc\Desktop\PIC1\DataSetup\STRING\gene_links_graph.dgl'
    data_path = r'C:\Users\xamuc\Desktop\PIC1\DataSetup\GTex\gene_tpm_whole_blood_graph.csv'
    ground_truth_path = r'C:\Users\xamuc\Desktop\PIC1\DataSetup\GTex\gene_tpm_brain_frontal_cortex_graph2.csv'

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
    seed = SetSeed(109757958)
    ES = EarlyStopper()
    model = GCN2(gt.shape[1]).to('cuda')
    train(model, g, data, gt)
    ES.PlotTitle = f'GCN {seed} - Loss per epoch'
    ES.PlotLoss()
    torch.save(model, rf'C:\Users\xamuc\Desktop\PIC1\DataSetup\Model\frontal_cortex_predictor.pth')

def PredMatrix(model, data):
    model.eval()
    preds = []
    for i in range(len(data)):
        x = data[i].unsqueeze(1)
        with torch.no_grad():
            pred = model(g, x).squeeze(0)
        preds.append(pred)
    preds = torch.stack(preds)
    return preds

#Evaluate
if 1:
    print("Loading model...")
    model = torch.load(r'C:\Users\xamuc\Desktop\PIC1\DataSetup\Model\frontal_cortex_predictor0.pth').to('cuda')

    pred_matrix = PredMatrix(model, data)
    mean_matrix = torch.stack([torch.mean(gt, dim=0)] * 159).to('cuda')

    E = Evaluate(pred_matrix, gt)
    E.std_measure()
    E.AvgResults()
    E = Evaluate(mean_matrix, gt)
    E.std_measure()
    E.AvgResults()

