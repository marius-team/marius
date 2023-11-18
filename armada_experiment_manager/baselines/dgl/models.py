import torch
import torch.nn as nn
import torch.nn.functional as F

import dgl
import dgl.function as fn

from dgl.nn.pytorch import NodeEmbedding, RelGraphConv, SAGEConv, GATConv



class GraphEmbeddingLayer(nn.Module):
    # For related examples, see the RelGraphEmbedLayer in
    # https://github.com/dmlc/dgl/blob/master/benchmarks/benchmarks/model_acc/bench_rgcn_ns.py
    # (very similar to bench_rgcn_homogeneous_ns.py)
    # https://github.com/dmlc/dgl/blob/master/benchmarks/benchmarks/model_speed/bench_rgcn_hetero_ns.py
    # https://github.com/dmlc/dgl/blob/master/examples/pytorch/rgcn-hetero/model.py
    # https://github.com/dmlc/dgl/blob/master/examples/pytorch/rgcn/model.py

    # Here we try to use https://docs.dgl.ai/api/python/nn.pytorch.html#nodeembedding-module

    # See https://github.com/dmlc/dgl/blob/master/examples/pytorch/rgcn/model.py for switching between
    # dgl.nn.NodeEmbedding and torch.nn.Embedding

    def __init__(self, num_nodes, embed_size, storage_device, backend='tensor'):
        super(GraphEmbeddingLayer, self).__init__()

        assert backend in ['tensor', 'torch_emb', 'dgl_sparse'], 'backend must be in [tensor, torch_emb, dgl_sparse]'

        self.backend = backend

        if backend == 'dgl_sparse':
            # meant to work for large scale node embs, supports partitioning to multiple GPUs,
            # SparseAdagrad and SparseAdam optimizers, e.g. optimizer = dgl.optim.SparseAdagrad([embs], lr=0.1)
            def initializer(emb):
                scale_factor = 0.001
                torch.nn.init.normal_(emb)
                emb = emb * scale_factor
                return emb

            self.node_embs = NodeEmbedding(num_nodes, embed_size, 'node_embs', init_func=initializer, device=storage_device)

        elif backend == 'torch_emb':
            self.node_embs = torch.nn.Embedding(num_nodes, embed_size, sparse=True, device=storage_device, dtype=torch.float32)

            scale_factor = 0.001
            torch.nn.init.normal_(self.node_embs.weight)
            self.node_embs.weight = nn.Parameter(self.node_embs.weight * scale_factor)

        else:
            node_embs = torch.Tensor(num_nodes, embed_size).to(torch.float32)
            torch.nn.init.normal_(node_embs)
            node_embs = node_embs * 0.001
            node_embs = node_embs.to(storage_device)
            self.node_embs = nn.Parameter(node_embs)

    def forward(self, node_ids, out_device):
        node_ids = node_ids.to(torch.int64)

        if self.backend == 'dgl_sparse':
            node_ids = node_ids.to(self.node_embs.weight.device)
            return self.node_embs(node_ids, out_device)

        elif self.backend == 'torch_emb':
            node_ids = node_ids.to(self.node_embs.weight.device)
            return self.node_embs(node_ids).to(out_device)

        else:
            node_ids = node_ids.to(self.node_embs.device)
            return self.node_embs[node_ids].to(out_device)



class DistMult(nn.Module):
    # Not sure if there are better ways to include a relation vector into the DotProduct examples that DGL has
    # e.g. see https://github.com/dmlc/dgl/blob/master/examples/pytorch/graphsage/train_sampling_unsupervised.py
    # https://docs.dgl.ai/tutorials/large/L2_large_link_prediction.html#sphx-glr-tutorials-large-l2-large-link-prediction-py

    def __init__(self, num_rels, emb_dim):
        super(DistMult, self).__init__()

        self.num_rels = num_rels
        self.emb_dim = emb_dim

        self.forward_rel_embs = nn.Parameter(torch.Tensor(num_rels, emb_dim))
        nn.init.ones_(self.forward_rel_embs)

        self.reverse_rel_embs = nn.Parameter(torch.Tensor(num_rels, emb_dim))
        nn.init.ones_(self.reverse_rel_embs)

        # nn.init.xavier_uniform_(self.rel_embs, gain=nn.init.calculate_gain('relu'))

    def forward(self, g, h, direction='forward'):
        assert direction in ['forward', 'reverse'], 'direction must be in [forward, reverse]'

        with g.local_scope():
            g.ndata['h'] = h

            if self.num_rels > 1:
                if direction == 'forward':
                    g.edata['w'] = self.forward_rel_embs[g.edata['etype'].to(torch.int64)]
                else:
                    g.edata['w'] = self.reverse_rel_embs[g.edata['etype'].to(torch.int64)]

                # Compute a new edge feature named 'score' by an element wise multiplication with the edge relation
                # weight vector, then a dot-product between the transformed source node feature
                # 'ht' and destination node feature 'h' (or the reverse).
                if direction == 'forward':
                    g.apply_edges(fn.u_mul_e('h', 'w', 'ht'))
                    g.apply_edges(fn.e_dot_v('ht', 'h', 'score'))
                else:
                    g.apply_edges(fn.v_mul_e('h', 'w', 'ht'))
                    g.apply_edges(fn.e_dot_u('ht', 'h', 'score'))

            else:
                g.apply_edges(fn.u_dot_v('h', 'h', 'score'))

            # u_dot_v returns a 1-element vector for each edge so you need to squeeze it.
            return g.edata['score'][:, 0]



class UniformChunkNegativeSampler(object):
    def __init__(self, num_chunks, num_uniform):
        super(UniformChunkNegativeSampler, self).__init__()

        self.num_chunks = num_chunks
        self.num_uniform = num_uniform

    def __call__(self, g, eids): #, canonical_etype):
        # this happens on the cpu I believe, but consequence of the cpu based edge data loader
        src, dst = g.find_edges(eids) #, etype=canonical_etype)

        batch_size = src.shape[0]
        num_chunks = torch.tensor(self.num_chunks, device=src.device)
        edges_per_chunk = torch.ceil(batch_size/num_chunks).to(torch.int32)

        src = src.repeat_interleave(self.num_uniform, 0)
        dst_neg = torch.randint(0, g.num_nodes(), (self.num_chunks, self.num_uniform), dtype=dst.dtype, device=dst.device)
        dst_neg = dst_neg.repeat_interleave(edges_per_chunk, 0)
        dst_neg = dst_neg.reshape((edges_per_chunk * self.num_chunks * self.num_uniform, ))
        dst_neg = dst_neg[:src.shape[0]]  # truncate if the last chunk doesn't have a full edges_per_chunk

        dst = dst.repeat_interleave(self.num_uniform, 0)
        src_neg = torch.randint(0, g.num_nodes(), (self.num_chunks, self.num_uniform), dtype=src.dtype, device=src.device)
        src_neg = src_neg.repeat_interleave(edges_per_chunk, 0)
        src_neg = src_neg.reshape((edges_per_chunk * self.num_chunks * self.num_uniform, ))
        src_neg = src_neg[:dst.shape[0]]  # truncate if the last chunk doesn't have a full edges_per_chunk

        src = torch.cat((src, src_neg), 0)
        dst = torch.cat((dst_neg, dst), 0)

        return src, dst



class NegativeSamplerForFilteredMrr(object):
    def __init__(self, corrupt='both'):
        super(NegativeSamplerForFilteredMrr, self).__init__()

        assert corrupt in ['both', 'dst'], 'corrupt must be in [both, dst]'

        self.corrupt = corrupt

    def __call__(self, g, eids):
        # this happens on the cpu I believe, but consequence of the cpu based edge data loader
        if self.corrupt == 'dst':
            src, _ = g.find_edges(eids)
            src = src.repeat_interleave(g.num_nodes(), 0)
            dst = torch.tile(torch.arange(g.num_nodes(), dtype=src.dtype, device=src.device), (eids.shape[0], ))
        else:
            src, dst = g.find_edges(eids)
            src = src.repeat_interleave(g.num_nodes(), 0)
            dst_neg = torch.tile(torch.arange(g.num_nodes(), dtype=dst.dtype, device=dst.device), (eids.shape[0],))

            dst = dst.repeat_interleave(g.num_nodes(), 0)
            src_neg = dst_neg

            src = torch.cat((src, src_neg), 0)
            dst = torch.cat((dst_neg, dst), 0)

        return src, dst





class GraphSage(nn.Module):
    def __init__(self, in_dim, h_dim, out_dim, n_layers, aggregator_type='mean', feat_drop=0.0, bias=True, norm=None):
        super(GraphSage, self).__init__()

        activation = F.relu

        self.layers = nn.ModuleList()

        # Initialization: bias zeros, xavier_uniform_ for weights with relu gain =>
        # Uniform(-a, a) with a = sqrt(2)*sqrt(6/(fan_in + fan_out)) with fan_in + fan_out = input_dim + output_dim

        if n_layers > 1:
            self.layers.append(SAGEConv(in_dim, h_dim, aggregator_type, feat_drop=feat_drop, bias=bias, norm=norm,
                                        activation=activation))
            for idx in range(n_layers - 2):
                self.layers.append(SAGEConv(h_dim, h_dim, aggregator_type, feat_drop=feat_drop, bias=bias, norm=norm,
                                            activation=activation))

            self.layers.append(SAGEConv(h_dim, out_dim, aggregator_type, feat_drop=feat_drop, bias=bias, norm=norm,
                                        activation=None))

        else:
            self.layers.append(SAGEConv(in_dim, out_dim, aggregator_type, feat_drop=feat_drop, bias=bias, norm=norm,
                                        activation=None))

    def forward(self, blocks, feats):
        # if blocks is None:
        #     # full graph training
        #     blocks = [self.g] * len(self.layers)

        h = feats
        for layer, b in zip(self.layers, blocks):
            h = layer(b, h)

        return h

# class GraphSage(nn.Module):
#     def __init__(self, in_feats, h_feats, num_classes, n_layers, aggregator_type='mean', feat_drop=0.0, bias=True, norm=None):
#         super(GraphSage, self).__init__()
#         self.conv1 = SAGEConv(in_feats, h_feats, aggregator_type="mean")
#         self.conv2 = SAGEConv(h_feats, num_classes, aggregator_type="mean")
#         self.h_feats = h_feats
#
#     def forward(self, mfgs, x):
#         # Lines that are changed are marked with an arrow: "<---"
#
#         h_dst = x[: mfgs[0].num_dst_nodes()]  # <---
#         h = self.conv1(mfgs[0], (x, h_dst))  # <---
#         h = F.relu(h)
#         h_dst = h[: mfgs[1].num_dst_nodes()]  # <---
#         h = self.conv2(mfgs[1], (h, h_dst))  # <---
#         return h



class GAT(nn.Module):
    def __init__(self, in_dim, h_dim, out_dim, num_heads, n_layers, feat_drop=0.0, attn_drop=0.0, negative_slope=0.2,
                 bias=True, allow_zero=False):
        super(GAT, self).__init__()

        activation = F.relu

        self.layers = nn.ModuleList()

        if n_layers > 1:
            self.layers.append(GATConv(in_dim, h_dim, num_heads, feat_drop=feat_drop, attn_drop=attn_drop,
                                       negative_slope=negative_slope, bias=bias, activation=activation,
                                       allow_zero_in_degree=allow_zero))
            for idx in range(n_layers - 2):
                self.layers.append(GATConv(h_dim, h_dim, num_heads, feat_drop=feat_drop, attn_drop=attn_drop,
                                           negative_slope=negative_slope, bias=bias, activation=activation,
                                           allow_zero_in_degree=allow_zero))

            self.layers.append(GATConv(h_dim, out_dim, num_heads, feat_drop=feat_drop, attn_drop=attn_drop,
                                       negative_slope=negative_slope, bias=bias, activation=None,
                                       allow_zero_in_degree=allow_zero))

        else:
            self.layers.append(GATConv(in_dim, out_dim, num_heads, feat_drop=feat_drop, attn_drop=attn_drop,
                                       negative_slope=negative_slope, bias=bias, activation=None,
                                       allow_zero_in_degree=allow_zero))

    def forward(self, blocks, feats):
        h = feats
        for layer, b in zip(self.layers, blocks):
            h = layer(b, h)
            h = torch.mean(h, dim=1) # average over the heads

        return h



class RGCN(nn.Module):
    # See # https://github.com/dmlc/dgl/blob/master/benchmarks/benchmarks/model_acc/bench_rgcn_ns.py

    def __init__(self, h_dim, out_dim, num_rels, num_bases=None, num_hidden_layers=1, dropout=0.0,
                 use_self_loop=True, low_mem=False, layer_norm=False):
        super(RGCN, self).__init__()

        self.h_dim = h_dim
        self.out_dim = out_dim
        self.num_rels = num_rels
        self.num_bases = num_bases
        self.num_hidden_layers = num_hidden_layers
        self.dropout = dropout
        self.use_self_loop = use_self_loop
        self.low_mem = low_mem
        self.layer_norm = layer_norm

        self.layers = nn.ModuleList()

        # TODO: match initialization for RelGraphConv in Marius
        # they initialize the relation matrices from Uniform(-a, a) with a = sqrt(2)*sqrt(6/(fan_in + fan_out)) with
        # fan_in + fan_out = (num_rels + input_dim) * output_dim
        # bias initialized to zero and self matrix initialized to Uniform(-a, a) with a as above and
        # fan_in + fan_out = input_dim + output_dim

        # i2h
        # self.layers.append(RelGraphConv(self.h_dim, self.h_dim, self.num_rels, "basis", self.num_bases, bias=True,
        #                                 activation=F.relu, self_loop=self.use_self_loop, low_mem=self.low_mem,
        #                                 dropout=self.dropout, layer_norm=self.layer_norm))

        # h2h
        for idx in range(self.num_hidden_layers):
            self.layers.append(RelGraphConv(self.h_dim, self.h_dim, self.num_rels, "basis", self.num_bases, bias=True,
                                            activation=F.relu, self_loop=self.use_self_loop, low_mem=self.low_mem,
                                            dropout=self.dropout, layer_norm=self.layer_norm))

        # h2o
        self.layers.append(RelGraphConv(self.h_dim, self.out_dim, self.num_rels, "basis", self.num_bases, bias=True,
                                        activation=None, self_loop=self.use_self_loop, low_mem=self.low_mem,
                                        layer_norm=self.layer_norm))

    def forward(self, blocks, feats):
        # if blocks is None:
        #     # full graph training
        #     blocks = [self.g] * len(self.layers)

        h = feats
        for layer, b in zip(self.layers, blocks):
            h = layer(b, h, b.edata['etype'], b.edata['norm'])

        return h



