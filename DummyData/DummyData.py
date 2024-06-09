import os
import random

import numpy as np
import torch
import dgl

import pandas as pd

def GenerateDir(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)
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

def GenerateGraph(src, dst, edges, k):
    g = dgl.graph((src, dst))
    g = dgl.to_bidirected(g)
    g.edata['w'] = torch.zeros(g.number_of_edges())
    g.edata['w'][g.edge_ids(src, dst)] = edges
    g.edata['w'][g.edge_ids(dst, src)] = edges
    dgl.save_graphs(project_dir + rf'\Dummy\Data\graph{k}.dgl', [g])
    return g

def GenerateData(g, weights, n, N=100):
    X = []
    Y = []
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
    df.to_csv(project_dir + rf'\Dummy\Data\data{n}.csv', index=False)

#Generate data sets
project_dir = rf'C:\Users\xamuc\Desktop\PIC1'
if 1:
    GenerateDir(project_dir + '\Dummy\Data')
    GenerateDir(project_dir + '\Dummy\Losses')
    #Graph
    src = [0,1,1,1,2,3,4]
    dst = [1,2,3,4,3,5,5]

    edges = torch.tensor([0.1, 0.3, 0.2, 0.4,0.05,0.15,0.25])
    g0 = GenerateGraph(src, dst, edges, 0)

    GenerateData(g0, [0.05, 0.35, 0.25,0.15,0.2,0], 0)
    GenerateData(g0, [0.0, 1.0, 0.0,0.0,0.0,0.0], 1)

    edges = torch.tensor([1.,1.,1.,1.,1.,1.,1.])
    g1 = GenerateGraph(src, dst, edges, 1)

    GenerateData(g1, [0.0, 1.0, 0.0,0.0,0.0,0.0], 2)

    #Fully connected
    src = []
    dst = []
    for i in range(6):
        for j in range(6):
            if i != j:
                src += [i]
                dst += [j]
    edges = torch.ones(30)
    g2 = GenerateGraph(src, dst, edges, 2)
    GenerateData(g2, [0.0, 1.0, 0.0,0.0,0.0,0.0], 3)
    GenerateData(g2, [0.05, 0.35, 0.25,0.15,0.2,0], 4)
