from torch_geometric.nn import SAGEConv, GATConv
import torch.nn.functional as F
from torch.nn import Parameter
import torch
import math


def compute_softmax(pos, neg):
    scores = torch.cat([pos.unsqueeze(1), neg.logsumexp(1, True)], 1)
    return torch.nn.functional.cross_entropy(scores, pos.new_zeros([], dtype=torch.int64).expand(pos.size(0)))


class Model(object):
    def __init__(self, node_embs, encoder, decoder, encoder_optimizer, decoder_optimizer, device):
        self.node_embs = node_embs

        if node_embs is not None:
            self.node_state = torch.zeros_like(node_embs)
        else:
            self.node_state = None
        self.encoder = encoder
        self.decoder = decoder
        self.encoder_optimizer = encoder_optimizer
        self.decoder_optimizer = decoder_optimizer
        self.device = device

    def zero_grad(self):

        if self.encoder_optimizer is not None:
            self.encoder_optimizer.zero_grad()
        if self.decoder_optimizer is not None:
            self.decoder_optimizer.zero_grad()

    def step(self, idx=None, node_grads=None):

        if self.encoder_optimizer is not None:
            self.encoder_optimizer.step()

        if self.decoder_optimizer is not None:
            self.decoder_optimizer.step()

        if idx is not None:
            idx = idx.to(self.node_embs.device)
            node_grads = node_grads.to(self.node_embs.device)

            curr_node_state = self.node_state[idx]
            node_state_update = node_grads.pow(2)
            curr_node_state.add_(node_state_update)
            node_grad_update = -.1 * (node_grads / (curr_node_state.sqrt().add_(1e-10)))

            self.node_state[idx] = curr_node_state
            self.node_embs[idx] += node_grad_update


class GAT(torch.nn.Module):
    def __init__(self, dims, compute_device, num_heads=10, concat=False, dropout=0.0):
        super(GAT, self).__init__()

        self.convs = torch.nn.ModuleList()
        self.compute_device = compute_device

        for i in range(len(dims) - 1):
            input_dim = dims[i]
            output_dim = dims[i + 1]
            self.convs.append(GATConv(input_dim, output_dim, heads=10, concat=concat, dropout=dropout).to(compute_device))

        self.reset_parameters()

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()

    def forward(self, x, nbrs, num_root):
        for i, conv in enumerate(self.convs):
            x = conv(x, nbrs)
            if i < len(self.convs) - 1:
                x = torch.relu(x)
        return x[:num_root]


class SAGE(torch.nn.Module):
    def __init__(self, dims, compute_device):
        super(SAGE, self).__init__()

        self.convs = torch.nn.ModuleList()
        self.compute_device = compute_device

        for i in range(len(dims) - 1):
            input_dim = dims[i]
            output_dim = dims[i + 1]
            self.convs.append(SAGEConv(input_dim, output_dim, root_weight=True).to(compute_device))

        self.reset_parameters()

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()

    def forward(self, x, nbrs, num_root):
        for i, conv in enumerate(self.convs):
            x = conv(x, nbrs)
            if i < len(self.convs) - 1:
                x = torch.relu(x)
        return x[:num_root]


class DistMultDecoder(torch.nn.Module):
    def __init__(self, num_relations, dim, device):
        super().__init__()
        self.rel_emb = Parameter(torch.Tensor(num_relations, dim).to(device))
        self.inv_rel_emb = Parameter(torch.Tensor(num_relations, dim).to(device))
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.ones_(self.rel_emb)
        torch.nn.init.ones_(self.inv_rel_emb)

    def forward(self, z_src, z_dst, edge_type, z_neg, corruption):

        num_pos = z_src.size(0)
        num_chunks = z_neg.size(0)
        num_per_chunk = math.ceil(num_pos / num_chunks)

        if corruption == "src":
            z_dst = z_dst * self.inv_rel_emb[edge_type]
        else:
            z_src = z_src * self.rel_emb[edge_type]

        if num_per_chunk != num_pos / num_chunks:
            new_size = num_per_chunk * num_chunks
            z_src = torch.nn.functional.pad(z_src, [0, 0, 0, new_size - num_pos])
            z_dst = torch.nn.functional.pad(z_dst, [0, 0, 0, new_size - num_pos])

        pos_scores = (z_src * z_dst).sum(-1)

        if corruption == "src":
            z_dst = z_dst.view([num_chunks, num_per_chunk, -1])
            neg_scores = z_dst.bmm(z_neg.transpose(-1, -2)).flatten(0, 1)
        else:
            z_src = z_src.view([num_chunks, num_per_chunk, -1])
            neg_scores = z_src.bmm(z_neg.transpose(-1, -2)).flatten(0, 1)

        return pos_scores, neg_scores


def select_encoder(config_args, compute_device):
    if config_args.model_encoder.upper() == "GRAPH_SAGE":
        return SAGE(config_args.dims, compute_device)
    elif config_args.model_encoder.upper() == "GAT":
        return GAT(config_args.dims, compute_device)
    elif config_args.model_encoder.upper() == "NONE":
        return None
    else:
        raise RuntimeError("Unrecognized encoder")


def select_decoder(config_args, compute_device):
    if config_args.model_decoder.upper() == "DISTMULT":
        return DistMultDecoder(config_args.num_relations, config_args.dims[-1], compute_device)
    else:
        raise RuntimeError("Unrecognized decoder")
