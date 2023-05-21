import torch
import torch.nn as nn
import torch.nn.functional as F

import dgl

"""
    Graph Transformer
    
"""

from graph_transformer_layer import GraphTransformerLayer
from scipy import sparse as sp


def laplacian_positional_encoding(g, pos_enc_dim):
    """
        Graph positional encoding v/ Laplacian eigenvectors
    """

    # Laplacian
    A = g.adjacency_matrix_scipy(return_edge_ids=False).astype(float)
    N = sp.diags(dgl.backend.asnumpy(g.in_degrees()).clip(1) ** -0.5, dtype=float)
    L = sp.eye(g.number_of_nodes()) - N * A * N

    # Eigenvectors with scipy
    #EigVal, EigVec = sp.linalg.eigs(L, k=pos_enc_dim+1, which='SR')
    EigVal, EigVec = sp.linalg.eigs(L, k=pos_enc_dim+1, which='SR', tol=1e-2) # for 40 PEs
    EigVec = EigVec[:, EigVal.argsort()] # increasing order
    g.ndata['lap_pos_enc'] = torch.from_numpy(EigVec[:,1:pos_enc_dim+1]).float()

    return g







class GraphTransformerNet(nn.Module):

    def __init__(self, in_feats, h_feats, num_classes, graph, num_heads, n_layers, layer_norm, batch_norm, residual, dropout, ffn, args, subsample_flag):
        super().__init__()

        in_dim_node = in_feats
        hidden_dim = h_feats
        out_dim = h_feats
        n_classes = num_classes
        #num_heads = 2
        #in_feat_dropout = net_params['in_feat_dropout']
        dropout = dropout
        #n_layers = 2

        #self.readout = net_params['readout']
        self.layer_norm = layer_norm
        self.batch_norm = batch_norm
        self.residual = residual
        #self.dropout = dropout
        self.n_classes = n_classes
  


        self.embedding_h = nn.Linear(in_dim_node, hidden_dim) # node feat is an integer
        self.pos_embedding = nn.Embedding(graph.number_of_nodes(), hidden_dim)
        


        self.layers = GraphTransformerLayer(hidden_dim, hidden_dim, num_heads, dropout, ffn, self.layer_norm, self.batch_norm, self.residual)
        self.n_layers = n_layers - 1 
        self.layers_out = GraphTransformerLayer(hidden_dim, out_dim, num_heads, dropout, ffn, self.layer_norm, self.batch_norm, self.residual)

        
        #self.MLP_layer = nn.Linear(2 *out_dim, n_classes)
        if args.homo:
            if subsample_flag:
                self.MLP_layer = nn.Linear(out_dim, n_classes)
            else:
                self.MLP_layer = nn.Linear(2* out_dim, n_classes)
        else:
            self.MLP_layer = nn.Linear(out_dim, n_classes)
            #self.MLP_layer = nn.Linear(out_dim *len(graph.canonical_etypes), n_classes)
        
        self.args = args

    def forward(self, g, h, subsample_flag):

        ##monitor energy
        energy = 0

        # input embedding
        h = self.embedding_h(h)


        h1 = h 


        if self.args.homo:
            for i in range(self.n_layers):
                h = self.layers(g, h)   
            h = self.layers_out(g, h)

            ## add original
            if not subsample_flag:
                h = torch.cat((h, h1), -1)

            h_out = self.MLP_layer(h)
        else:
            h_all = []
            for relation in g.canonical_etypes:
                h_0 = h 
                for i in range(self.n_layers):
                    h_0 = self.layers(g[relation], h_0)

        
                h_0 = self.layers_out(g[relation], h_0)
            
                h_all.append(torch.nan_to_num(h_0))
        
            ## add original
            h_all.append(h1)

            h_rst = torch.max(torch.stack(h_all), dim=0)[0]
        
            h_out = self.MLP_layer(h_rst)

        return h_out
    
    



        
