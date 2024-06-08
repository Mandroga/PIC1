import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn.pytorch import GraphConv, AvgPooling, SumPooling


class GCN0(nn.Module):
    def __init__(self, num_classes):
        super(GCN0, self).__init__()

        self.conv = GraphConv(1, 1000)
        self.pool = AvgPooling()
        self.lin0 = nn.Linear(1000, 100)
        self.lin1 = nn.Linear(100, num_classes)

    def forward(self, g, in_feat):
        h = self.conv(g, in_feat)
        h = F.relu(h)
        h = self.pool(g,h)
        h = F.relu(h)
        h = self.lin0(h)
        h = F.relu(h)
        h = self.lin1(h)
        h = F.relu(h)

        return h

class GCN1(nn.Module):
    def __init__(self, num_classes):
        super(GCN1, self).__init__()

        self.conv0 = GraphConv(1, 1000)
        self.conv1 = GraphConv(1000,1)
        self.lin1 = nn.Linear(15840, 100)
        self.lin2 = nn.Linear(100, num_classes)

        nn.init.uniform_(self.conv0.weight) # todas as layers, nao treinar prever só
        nn.init.uniform_(self.conv1.weight)
        nn.init.uniform_(self.lin1.weight)
        nn.init.uniform_(self.lin2.weight)
    def forward(self, g, in_feat):
       # torch.Size([15840, 1])
        h = self.conv0(g, in_feat)
        # torch.Size([15840, 1000])
        h = F.relu(h)
        h = self.conv1(g,h)
         # torch.Size([15840, 1])
        h = F.relu(h)
        #h = h.reshape((1,-1))
        h = h.T
        h = self.lin1(h)
        h = F.relu(h)
        h = self.lin2(h)
        h = F.relu(h)

        return h

    #Fazer Graph Conv e depois ir buscar os valores que quero aos nós

class GCN2(nn.Module):
    def __init__(self, num_classes):
        super(GCN2, self).__init__()

        self.conv0 = GraphConv(1, 100)
        self.conv1 = GraphConv(100,10)
        self.conv2 = GraphConv(10, 1)
    def forward(self, g, in_feat):
        h = self.conv0(g, in_feat)
        h = F.leaky_relu(h)
        h = self.conv1(g,h)
        h = F.relu(h)
        h = self.conv2(g, h)
        h = F.relu(h)
        #h = self.conv3(g, h)
        #h = F.relu(h)
        return h

class DummyGCN1(nn.Module):
    def __init__(self, h1, h2 , h3):
        super(DummyGCN1, self).__init__()

        self.conv0 = GraphConv(1, h1)
        self.conv1 = GraphConv(h1, 1)
        self.lin0 = nn.Linear(6, h2)
        self.lin2 = nn.Linear(h2, h3)
        self.lin3 = nn.Linear(h3, 1)

    def forward(self, g, in_feat):
        h = self.conv0(g, in_feat)
        h = F.leaky_relu(h)
        h = self.conv1(g,h)
        h = F.leaky_relu(h)
        h = torch.transpose(h,0,2)
        h = self.lin0(h)
        h = F.leaky_relu(h)
        h = self.lin2(h)
        h = F.leaky_relu(h)
        h = self.lin3(h)
        h = F.leaky_relu(h)
        h = h.squeeze(0)
        return h

class DummyGCN2(nn.Module):
    def __init__(self, h1, h2, h3):
        super(DummyGCN2, self).__init__()

        self.conv0 = GraphConv(1, h1)
        self.conv1 = GraphConv(h1,h2)
        self.conv2 = GraphConv(h2, h3)
        self.conv3 = GraphConv(h3, 1)
    def forward(self, g, in_feat):
       # print(in_feat.shape, 'in')
        h = self.conv0(g, in_feat)
       # print(h.shape, 'conv0')
        h = F.leaky_relu(h)
        h = self.conv1(g,h)
       # print(h.shape, 'conv1')
        h = F.leaky_relu(h)
        h = self.conv2(g, h)
      #  print(h.shape, 'conv2')
        h = F.leaky_relu(h)
        h = self.conv3(g, h)
       # print(h.shape, 'conv3')
        h = F.leaky_relu(h)
        h = h[1]
       # print(h.shape, 'out')
        return h

class DummyGCN3(nn.Module):
    def __init__(self, h1):
        super(DummyGCN3, self).__init__()

        self.conv0 = GraphConv(1, h1)
        self.conv1 = GraphConv(h1,1)

    def forward(self, g, in_feat):
        h = self.conv0(g, in_feat)
        h = F.leaky_relu(h)
        h = self.conv1(g,h)
        h = F.leaky_relu(h)
        h = h[1]
        return h