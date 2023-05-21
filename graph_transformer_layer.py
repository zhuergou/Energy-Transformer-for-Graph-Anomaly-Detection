import torch
import torch.nn as nn
import torch.nn.functional as F

import dgl
import dgl.function as fn
import numpy as np

from torch.autograd import Variable

"""
    Graph Transformer Layer
    
"""

"""
M
    Util functions
"""
def src_dot_dst(src_field, dst_field, out_field):
    def func(edges):
        return {out_field: (edges.src[src_field] * edges.dst[dst_field]).sum(-1, keepdim=True)}
    return func

def scaled_exp(field, scale_constant):
    def func(edges):
        # clamp for softmax numerical stability
        return {field: torch.exp((edges.data[field] / scale_constant).clamp(-5, 5))}
        #return {field: torch.exp((edges.data[field] / scale_constant))}

    return func

    
def custom_message_func(src_f1_field, edge_field, out_field, src_f2_field):
    def func(edges):
        return {out_field: edges.src[src_f1_field]*edges.data[edge_field]/edges.src[src_f2_field]}
    return func



"""
    Single Attention Head
"""

class MultiHeadAttentionLayer(nn.Module):
    def __init__(self, in_dim, out_dim, num_heads, use_bias):
        super().__init__()
        
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.in_dim = in_dim
        self.beta = nn.Parameter(nn.init.constant_(torch.empty(1), np.sqrt(self.out_dim))) 
        
        if use_bias:
            self.Q = nn.Parameter(nn.init.normal_(torch.Tensor(in_dim, num_heads, out_dim), std=0.01))
            self.K = nn.Parameter(nn.init.normal_(torch.Tensor(in_dim, num_heads, out_dim), std=0.01))
        else:
            self.Q = nn.Parameter(nn.init.normal_(torch.Tensor(in_dim, num_heads, out_dim), std=0.01))
            self.K = nn.Parameter(nn.init.normal_(torch.Tensor(in_dim, num_heads, out_dim), std=0.01))
        
    
    def propagate_attention(self, g):
        eids = g.edges()
        # Compute attention score
        #g.apply_edges(src_dot_dst('K_h', 'Q_h', 'score')) #, edges)
        g.apply_edges(src_dot_dst('K_h', 'Q_h', 'score'), eids)
        g.apply_edges(scaled_exp('score', self.beta), eids)

        # Send weighted values to target nodes
        g.send_and_recv(eids, fn.src_mul_edge('F1', 'score', 'V_1'), fn.sum('V_1', 'T1'))
        g.send_and_recv(eids, fn.copy_edge('score', 'score'), fn.sum('score', 'z'))

       
        g.apply_edges(src_dot_dst('Q_h', 'K_h', 'score_2'), eids)
        g.apply_edges(scaled_exp('score_2', self.beta), eids)

        g.send_and_recv(eids, custom_message_func('F2', 'score_2', 'V_2', 'z'), fn.sum('V_2', 'T2'))
         

    

    def manual_energy(self, g):
        all_edge = g.ndata['z'].squeeze()  ## B * H
        A2 = torch.log(all_edge)
        energy = (self.beta*A2).sum(axis = -1)
        return energy


    def forward(self, g, h):
        
        #print(self.Q)
        Q_h = torch.einsum("bd, dhz-> bhz", h, self.Q)
        K_h = torch.einsum("bd, dhz-> bhz", h, self.K)
        
        
        g.ndata['Q_h'] = Q_h
        g.ndata['K_h'] = K_h

        F1 = torch.einsum("dzh,bhz->bhd", self.Q.permute(0,2,1), K_h)
        F2 = torch.einsum("dzh,bhz->bhd", self.K.permute(0,2,1), Q_h)

        g.ndata['F1'] = F1
        g.ndata['F2'] = F2


        self.propagate_attention(g)        


        head_out_1 = g.ndata['T1']/g.ndata['z']   ## B* H * in_dim
        head_out_2 = g.ndata['T2']                ## B* H * in_dim
        
        final = torch.sum(head_out_1 + head_out_2, axis=1)   ## B* in_dim
        #final = torch.sum(head_out_1, axis=1)

        ## compute the energy
        energy = self.manual_energy(g)
        
        return final, energy
    

class GraphTransformerLayer(nn.Module):
    """
        Param: 
    """
    def __init__(self, in_dim, out_dim, num_heads, dropout=0.0, ffn=4, layer_norm=False, batch_norm=False, residual=False, use_bias=False):
        super().__init__()

        self.in_channels = in_dim
        self.out_channels = out_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.residual = residual
        self.layer_norm = layer_norm        
        self.batch_norm = batch_norm
        self.beta_2 = nn.Parameter(nn.init.constant_(torch.empty(1), 1)) 
        
        self.attention = MultiHeadAttentionLayer(in_dim, out_dim, num_heads, use_bias)
        
        self.O = nn.Linear(out_dim, out_dim)

        if self.layer_norm:
            self.layer_norm1 = nn.LayerNorm(out_dim)
            
        if self.batch_norm:
            self.batch_norm1 = nn.BatchNorm1d(out_dim)
        
        # FFN
        #self.FFN_layer1 = nn.Linear(out_dim, out_dim*2)
        #self.FFN_layer2 = nn.Linear(out_dim*2, out_dim)
        #self.W = nn.Parameter(nn.init.normal_(torch.Tensor(out_dim, out_dim*2), mean=0.0, std=0.01))
        self.W = nn.Parameter(nn.init.uniform_(torch.Tensor(out_dim, out_dim*ffn), a=-1.0/np.sqrt(out_dim), b=1.0/np.sqrt(out_dim)))

        if self.layer_norm:
            self.layer_norm2 = nn.LayerNorm(out_dim)
            
        if self.batch_norm:
            self.batch_norm2 = nn.BatchNorm1d(out_dim)
        
    


    def forward(self, g, h):
        if self.layer_norm:
            h = self.layer_norm1(h)
            
        if self.batch_norm:
            h = self.batch_norm1(h)
        
        h_in1 = h # for first residual connection
        h2 = h


        #h = Variable(h, requires_grad=True)
        
        # multi-head attention out
        attn_out, energy = self.attention(g, h)
        h_out = attn_out



        h = h_out
        #h = F.dropout(h, self.dropout, training=self.training)
        
        #h = self.O(h)
        
        h2 = torch.matmul(h2, self.W)
        h2 = F.relu(h2)
        h2 = F.dropout(h2, self.dropout, training=self.training)
        h2 = torch.matmul(h2, self.W.t())
        h = h + h2


        
    

        if self.residual:
            h = h_in1 + h # residual connection
        

        

        return h
        
    def __repr__(self):
        return '{}(in_channels={}, out_channels={}, heads={}, residual={})'.format(self.__class__.__name__,
                                             self.in_channels,
                                             self.out_channels, self.num_heads, self.residual)
