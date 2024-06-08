import dgl
import torch
import random
import pandas as pd
import numpy as np
def connecting_nodes(graph, node_id):
    edges_from_node = (graph.edges()[0] == node_id)
    node_edges = []
    # Access edge values connected to the node
    for edge_id in edges_from_node.nonzero():
        edge_id = edge_id.item()
        value = graph.edata['w'][edge_id].item()
        src, dst = graph.find_edges(edge_id)
        node_edges.append([dst.item(), value])
        #print(f"Edge ({src.item()}, {dst.item()}): {value.item()}")
    return node_edges

n = 0
#Graph
if 0:
    k = 1
    src = [0,1,1,1,2,3,4]
    dst = [1,2,3,4,3,5,5]

#Fully connected Graph
if 1:
    k = 2
    src = []
    dst = []
    for i in range(6):
        for j in range(6):
            if i != j:
                src += [i]
                dst += [j]
    edges = torch.ones(30)


#data set 1
if 0:
    n = 1
    weights = [0.05, 0.35, 0.25,0.15,0.2,0]
    edges = torch.tensor([0.1, 0.3, 0.2, 0.4,0.05,0.15,0.25])
#data set 2
if 0:
    n = 2
    weights = [0.0, 1.0, 0.0,0.0,0.0,0.0]
    edges = torch.tensor([0.1, 0.3, 0.2, 0.4,0.05,0.15,0.25])
#data set 3
if 0:
    n = 3
    weights = [0.0, 1.0, 0.0,0.0,0.0,0.0]
    edges = torch.tensor([1.,1.,1.,1.,1.,1.,1.])
#data set 4
if 0:
    n = 4
    weights = [0.0, 1.0, 0.0,0.0,0.0,0.0]

#data set 5
if 1:
    n = 5
    weights = [0.05, 0.35, 0.25,0.15,0.2,0]

#def graph
if 1:
    g = dgl.graph((src, dst))
    g = dgl.to_bidirected(g)
    g.edata['w'] = torch.zeros(g.number_of_edges())
    g.edata['w'][g.edge_ids(src, dst)] = edges
    g.edata['w'][g.edge_ids(dst, src)] = edges
    dgl.save_graphs(rf'C:\Users\xamuc\Desktop\PIC1\DataSetup\Dummy\dummygraph{k}.dgl', [g])


X = []
Y = []

N = 100
for i in range(N):
    features = torch.tensor([random.randint(0, 1000) for _ in range(6)], dtype=torch.float32)
    g.ndata['feat'] = features
    y = 0
    for n1 in range(6):
        contribution = 0
        contribution += features[n1]
        connected_nodes = connecting_nodes(g, n1)
        for edge in connected_nodes:
            n2 = edge[0]
            val = edge[1]
            contribution += features[n2] * val
        y += contribution * weights[n1]

    X += [features.tolist()]
    Y += [y]

X = np.array(X)
Y = np.array(Y)

df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(X.shape[1])])
df['Y'] = Y

# Save the DataFrame to a CSV file
df.to_csv(rf'C:\Users\xamuc\Desktop\PIC1\DataSetup\Dummy\data{n}.csv', index=False)