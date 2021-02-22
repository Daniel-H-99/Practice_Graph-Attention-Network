import torch
import torch.nn as nn 
import torch.nn.functional as F 
from layer import GraphConvolutionLayer, GraphAttentionLayer, SparseGraphConvolutionLayer, SparseGraphAttentionLayer

# TODO step 1.
class GCN(nn.Module):
    def __init__(self, args, nfeat, nhid, nclass, dropout):
        super(GCN, self).__init__()
        self.layer1 = GraphConvolutionLayer(args, nfeat, nhid, dropout)
        self.layer2 = GraphConvolutionLayer(args, nhid, nclass, dropout)
        self.ReLU = nn.ReLU()
        self.LogSoftmax = nn.LogSoftmax(dim=1)

    def forward(self, x, adj):
        output = self.ReLU(self.layer1(x, adj))
        output = self.LogSoftmax(self.layer2(output, adj))
        return output
    
# TODO step 2.
class GAT(nn.Module):
    def __init__(self, args, nfeat, nhid, nclass, dropout, alpha, nheads):
        super(GAT, self).__init__()
        self.layer1 = GraphAttentionLayer(args, nfeat, nhid, nheads, dropout, alpha)
        self.layer2 = GraphAttentionLayer(args, nhid, nclass, 1, dropout, alpha)
        self.ReLU = nn.ReLU()
        self.LogSoftmax = nn.LogSoftmax(dim=1)        

    def forward(self, x, adj):
        output = self.ReLU(self.layer1(x, adj))
        output = self.LogSoftmax(self.layer2(output, adj))
        return output

# TODO step 3.
class SpGCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(SpGCN, self).__init__()
        pass

    def forward(self, x, adj):
        pass

class SpGAT(nn.Module):
    def __init__(self, args, nfeat, nhid, nclass, dropout, alpha, nheads):
        super(SpGAT, self).__init__()
        self.layer1 = SparseGraphAttentionLayer(args, nfeat, nhid, nheads, dropout, alpha)
        self.layer2 = SparseGraphAttentionLayer(args, nhid, nclass, 1, dropout, alpha)
        self.ReLU = nn.ReLU()
        self.LogSoftmax = nn.LogSoftmax(dim=1)             

    def forward(self, x, adj):
        output = self.ReLU(self.layer1(x, adj))
        output = self.LogSoftmax(self.layer2(output, adj))
        return output