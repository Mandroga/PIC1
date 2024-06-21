import os
import random

import numpy as np
import torch
import dgl
import networkx as nx
import matplotlib.pyplot as plt

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
def GenerateData(g, weights, feature_max_val, N=100):
    X = []
    Y = []
    for i in range(N):
        features = torch.tensor([random.randint(0, feature_max_val) for _ in range(6)], dtype=torch.float32)
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

    return df
def GenerateData2(g, weights, feature_max_val, N=100):
    X = []
    Y = []
    for i in range(N):
        features = torch.tensor([random.uniform(0.001, feature_max_val) for _ in range(6)], dtype=torch.float32)
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

    return df
def PlotDGLGraph(graph):
    nx_graph = dgl.to_networkx(graph, node_attrs=['feat'], edge_attrs=['w'])
    has_labels = True
    has_colors = True
    has_edgedata = False
    # Extract node labels (features)
    #try: node_labels = {node: f"{node} - {graph.ndata['feat'][node].tolist()}" for node in nx_graph.nodes()}
    try: node_labels = {node: f"{node}" for node in nx_graph.nodes()}
    except: has_labels = False
    try: node_colors = ['red' if graph.ndata['label'][node].item()==1 else 'blue' for node in nx_graph.nodes]
    except: has_colors = False
    # Extract edge labels (values)
    try:
        edge_labels = {}
        for src, dst in nx_graph.edges():
            edge_ids = graph.edge_ids(src, dst)
            edge_labels[(src, dst)] = round(graph.edata['w'][edge_ids].item(),1)
    except: has_edgedata = False
    # Plot the graph
    pos = nx.spring_layout(nx_graph)  # Position nodes using the spring layout algorithm
    if has_labels and has_colors and has_edgedata:
        nx.draw(nx_graph, pos, with_labels=True, labels=node_labels, node_color=node_colors, node_size=1500, font_size=12)
        nx.draw_networkx_edge_labels(nx_graph, pos, edge_labels=edge_labels, font_color='red')
    elif has_labels and has_edgedata:
        nx.draw(nx_graph, pos, with_labels=True, labels=node_labels, node_size=2000,
                font_size=9)
        nx.draw_networkx_edge_labels(nx_graph, pos, edge_labels=edge_labels, font_color='red')
    elif has_labels:
        nx.draw(nx_graph, pos, with_labels=True, labels=node_labels, node_size=2000,
                font_size=9)
    elif has_edgedata:
        nx.draw(nx_graph, pos,  node_size=1500,font_size=12)
        nx.draw_networkx_edge_labels(nx_graph, pos, edge_labels=edge_labels, font_color='red')
    else:
        nx.draw(nx_graph, pos, node_size=1500, font_size=12)
    plt.show()

#Generate data sets
project_dir = rf'C:\Users\xamuc\Desktop\PIC1'
if 1:
    GenerateDir(project_dir + '\Dummy\Data')
    GenerateDir(project_dir + '\Dummy\Losses')
    data_dir = project_dir + '\Dummy\Data'
    #Graph
    src = [0,1,1,1,2,3,4]
    dst = [1,2,3,4,3,5,5]

    edges = torch.tensor([0.1, 0.3, 0.2, 0.4,0.05,0.15,0.25])
    g0 = GenerateGraph(src, dst, edges, 0)

    GenerateData(g0, [0.05, 0.35, 0.25,0.15,0.2,0],1000).to_csv(data_dir + rf'\data0.csv', index=False)
    #PlotDGLGraph(g0)
    GenerateData(g0, [0.0, 1.0, 0.0,0.0,0.0,0.0],1000).to_csv(data_dir + rf'\data1.csv', index=False)

    df = GenerateData2(g0, [0.0, 1.0, 0.0,0.0,0.0,0.0],100)
    df['Y'] = df['Y']/100
    print(df)
    df.to_csv(data_dir + rf'\data5.csv', index=False)

    df = GenerateData2(g0, [0.05, 0.35, 0.25,0.15,0.2,0], 1)
    df['Y'] = df['Y'] / max(df['Y'])
    df.to_csv(data_dir + rf'\data6.csv', index=False)

    df = GenerateData2(g0, [0.05, 0.35, 0.25, 0.15, 0.2, 0], 1)
    df['Y'] = df['Y'] / df['Y'].mean()
    df.to_csv(data_dir + rf'\data7.csv', index=False)

    df = GenerateData2(g0, [0.0, 1.0, 0.0, 0.0, 0.0, 0.0], 1)
    print(df)
    df['Y'] = df['Y'] / max(df['Y'])
    df.to_csv(data_dir + rf'\data8.csv', index=False)

    df = GenerateData2(g0, [0.0, 1.0, 0.0, 0.0, 0.0, 0.0], 1)
    df['Y'] = df['Y'] / df['Y'].mean()
    df.to_csv(data_dir + rf'\data9.csv', index=False)

    edges = torch.tensor([1.,1.,1.,1.,1.,1.,1.])
    g1 = GenerateGraph(src, dst, edges, 1)

    GenerateData(g1, [0.0, 1.0, 0.0,0.0,0.0,0.0],1000).to_csv(data_dir + rf'\data2.csv', index=False)
    #PlotDGLGraph(g1)
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
    GenerateData(g2, [0.0, 1.0, 0.0,0.0,0.0,0.0],1000).to_csv(data_dir + rf'\data3.csv', index=False)
    GenerateData(g2, [0.05, 0.35, 0.25,0.15,0.2,0],1000).to_csv(data_dir + rf'\data4.csv', index=False)
    PlotDGLGraph(g2)