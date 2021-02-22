
import numpy as np 
import torch 
import torch.nn as nn 
import torch.nn.functional as F 

# TODO step 1. 
class GraphConvolutionLayer(nn.Module):
    def __init__(self, args, in_features, out_features, dropout):
        super(GraphConvolutionLayer,self).__init__()
        self.kernel = nn.Linear(in_features, out_features)
        self.dropout = nn.Dropout(p=dropout)
        nn.init.xavier_normal(self.kernel.weight)
    def forward(self, input, adj):
        # input: (N x in_features)
        # output: (N x out_features)
        
        # apply kernel : (N x out_features)
        output = self.dropout(self.kernel(input))
 
        # aggregation
        output = self.dropout(torch.mm(adj, output))
        return output

# TODO step 2. 
class GraphAttentionLayer(nn.Module):
    """multihead attention """ 
    def __init__(self, args, in_features, out_features, num_heads, dropout, alpha, concat=True):
        super(GraphAttentionLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features // num_heads
        self.num_heads = num_heads
        self.concat = concat
        self.kernel = nn.Linear(in_features, num_heads * self.out_features)
        self.attn_weights = torch.FloatTensor(num_heads, self.out_features, 2).requires_grad_().to(args.device)
        self.dropout = nn.Dropout(p=dropout)
        self.LeakyReLU = nn.LeakyReLU(alpha)
        nn.init.xavier_normal(self.kernel.weight)
        nn.init.xavier_normal(self.attn_weights)
        
    def forward(self, input, adj):
        # input: (N x in_features)
        # output: (N x N)
        # apply kernel: (N x num_heads * out_features)
        output = self.dropout(self.kernel(input))
        # reshape to (num_heads x N x out_features)
        output = output.view(-1, self.num_heads, self.out_features).transpose(0, 1)
        # caculate attention
        adj = adj.unsqueeze(0).expand(self.num_heads, -1, -1)
        attn = self.dropout(torch.bmm(output, self.attn_weights))   # (num_heads x N x 2)
        attn = self.LeakyReLU(attn[:,:,0].unsqueeze(2) + attn[:, :, 1].unsqueeze(1))   # (num_heads x N x N)
        attn = attn.masked_fill(adj <= 0, -1e9)
        attn = nn.Softmax(dim=2)(attn)
        attn = attn.masked_fill(adj <= 0, 0)   # (num_heads x N x N)
        # aggregation
        output = self.dropout(torch.bmm(attn, output))
        output = output.transpose(0, 1)   # (N x num_heads x out_features)
        
        if self.concat:
            output = output.contiguous().view(-1, self.num_heads * self.out_features)
        else:
            output = output.mean(dim=1)
        return output
   
        
# TODO step 3.
class SparsemmFunction(torch.autograd.Function):
    """ for only sparse region backpropataion layer."""
    @staticmethod
    def forward(ctx, indices, values, shape, b):
        a = torch.sparse_coo_tensor(indices, values, shape)
        ctx.rowsize = shape[0]
        ctx.save_for_backward(a, b)
        return torch.mm(a.cpu(), b).cuda()

    @staticmethod
    def backward(ctx, grad_output):
        a, b = ctx.saved_tensors
        if ctx.needs_input_grad[1]:
            grad_a = torch.mm(grad_output, b.transpose(0, 1))
            grad_values = grad_a[N * a._indices()[0] + a._indices()[1]]
        else:
            grad_values = None
        if ctx.needs_input_grad[3]:
            grad_b = torch.mm(a.transpose(0, 1), grad_output)
        else:
            grad_b = None
        return None, grad_values, None, grad_b

class Sparsemm(nn.Module):
    def forward(self, indices, values, shape, b):
        return SparsemmFunction.apply(indices, values, shape, b)


class SparseGraphConvolutionLayer(nn.Module):
    def __init__(self, in_features, out_features, dropout):
        super(SparseGraphConvolutionLayer,self).__init__()
        pass
    def forward(self, input, adj):
        pass

class SparseGraphAttentionLayer(nn.Module):
    """multihead attention """ 
    def __init__(self, args, in_features, out_features, num_heads, dropout, alpha, concat=True):
        super(SparseGraphAttentionLayer, self).__init__()
        self.args = args
        self.in_features = in_features
        self.out_features = out_features // num_heads
        self.num_heads = num_heads
        self.concat = concat
        self.kernel = nn.Linear(in_features, num_heads * self.out_features)
        self.attn_weights = torch.FloatTensor(num_heads, self.out_features, 2).requires_grad_().to(args.device)
        self.dropout = nn.Dropout(p=dropout)
        self.LeakyReLU = nn.LeakyReLU(alpha)
        self.Sparsemm = Sparsemm()
        nn.init.xavier_normal(self.kernel.weight)
        nn.init.xavier_normal(self.attn_weights)
    
    def forward(self, input, adj):
        # input: (N x in_features)
        # output: (N x N)
        N = input.shape[0]
        # apply kernel: (N x num_heads * out_features)
        output = self.dropout(self.kernel(input))
        # reshape to (num_heads x N x out_features)
        output = output.view(-1, self.num_heads, self.out_features).transpose(0, 1)
        # construct expanded adjency matrix with block diagonal of each head
        adj = torch.cat([adj.unsqueeze(0).unsqueeze(0).repeat(self.num_heads, 1, 1, 1), torch.zeros(self.num_heads, self.num_heads, adj.shape[0], adj.shape[0]).to(self.args.device)], dim=1).view(self.num_heads + 1, self.num_heads, N, N)[:self.num_heads, :, :, :]
        adj = adj.permute(0, 2, 1, 3).contiguous().view(self.num_heads * N, self.num_heads * N)
        adj_indices = adj.nonzero().transpose(0, 1)
        # caculate attention
        attn = self.dropout(torch.bmm(output, self.attn_weights).view(self.num_heads * N, 2))
        attn_values = self.LeakyReLU(attn[adj_indices[0], 0] + attn[adj_indices[1], 1])
        attn_values = attn_values.exp()
        attn = (torch.mm(torch.sparse_coo_tensor(adj_indices, attn_values, adj.shape).cpu(), torch.eye(adj.shape[0])) / self.Sparsemm(adj_indices, attn_values, adj.shape, torch.ones(N * self.num_heads, 1)).cpu()).cuda()
        attn_values = attn[adj_indices]
        # aggretation
        output = self.dropout(self.Sparsemm(adj_indices, attn_values, adj.shape, output))
        output = output.view(self.num_heads, N, self.out_features).transpose(0, 1)
        
        if self.concat:
            output = output.contiguous().view(-1, self.num_heads * self.out_features)
        else:
            output = output.mean(dim=1)
        return output        