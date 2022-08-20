from torch_geometric.loader.utils import filter_data, to_csc
import torch
import sys
import math


def get_mapping(edges, neg_src, neg_dst):

    root_nodes, mapping = torch.unique(torch.cat([edges[0],
                                                  edges[-1],
                                                  neg_src.flatten(0, 1),
                                                  neg_dst.flatten(0, 1)]),
                                       return_inverse=True)

    offset = 0
    src_mapping = mapping[offset:edges.size(1)]

    offset += edges.size(1)
    dst_mapping = mapping[offset:offset + edges.size(1)]

    offset += edges.size(1)
    src_neg_mapping = mapping[offset:offset + neg_src.flatten(0, 1).size(0)]

    offset += neg_src.flatten(0, 1).size(0)
    dst_neg_mapping = mapping[offset:offset + neg_dst.flatten(0, 1).size(0)]

    return root_nodes, src_mapping, dst_mapping, src_neg_mapping, dst_neg_mapping


def edges_to_unique_edge_ids(edges, num_nodes):
    """
    Convert triple to a unique int64 id
    """
    src_stride = 1
    dst_stride = num_nodes
    rel_stride = num_nodes * num_nodes
    return src_stride * edges[0] + rel_stride * edges[1] + dst_stride * edges[2]


def edge_ids_to_edges(edge_ids, num_nodes):
    """
    Convert unique int64 id to a triple
    """
    src_ids = edge_ids % num_nodes
    edge_ids = torch.div(edge_ids, num_nodes, rounding_mode='floor')
    dst_ids = edge_ids % num_nodes
    rel_ids = torch.div(edge_ids, num_nodes, rounding_mode='floor')
    return torch.stack([src_ids, rel_ids, dst_ids])


# Only can perform CPU neighbor sampling
class NeighborSampler(object):
    def __init__(self, data):
        self.data = data
        self.colptr, self.row, self.perm = to_csc(data)

    def sample_nbrs(self, node_ids, num_neighbors=None, replacement=True):
        if num_neighbors is None:
            num_neighbors = [-1]

        dev = node_ids.device

        node, row, col, edge = torch.ops.torch_sparse.neighbor_sample(
            self.colptr,
            self.row,
            node_ids.to(torch.device("cpu")),
            num_neighbors,
            replacement,
            True
        )
        output_data = filter_data(self.data, node, row, col, edge, self.perm)
        return node.to(dev), output_data.to(dev)


class NegativeSampler(object):
    def __init__(self, data, num_chunks, num_negs, degree_fraction, filtered):
        self.data = data
        self.num_chunks = num_chunks
        self.num_negs = num_negs
        self.degree_fraction = degree_fraction
        self.filtered = filtered

        if self.filtered:
            all_edges_nodes = torch.cat([data.train_split, data.valid_split, data.test_split], 1)
            all_edges_types = torch.cat([data.train_edge_type, data.valid_edge_type, data.test_edge_type], 0)
            all_edges = torch.stack([all_edges_nodes[0], all_edges_types, all_edges_nodes[1]], 0)
            sorted_edge_ids, _ = torch.sort(edges_to_unique_edge_ids(all_edges, data.num_nodes))
            self.sorted_edge_ids = torch.cat([sorted_edge_ids, torch.tensor([sys.maxsize], device=sorted_edge_ids.device)])

    def sample(self, batch_edges):

        neg_src = None
        neg_dst = None
        src_neg_filter = None
        dst_neg_filter = None

        if not self.filtered:
            num_uni = int(self.num_negs * (1 - self.degree_fraction))
            num_deg = self.num_negs - num_uni

            num_pos = batch_edges.size(1)
            num_per_chunk = math.ceil(num_pos / self.num_chunks)

            uni_neg_src = torch.randint(self.data.num_nodes, size=[self.num_chunks, num_uni], device=batch_edges.device)
            uni_neg_dst = torch.randint(self.data.num_nodes, size=[self.num_chunks, num_uni], device=batch_edges.device)

            if num_deg > 0:
                deg_neg_src = torch.randint(batch_edges.size(1), size=[self.num_chunks, num_deg], device=batch_edges.device)
                chunk_ids = deg_neg_src.div(num_per_chunk).view([self.num_chunks, -1])
                inv_mask = chunk_ids - torch.arange(0, self.num_chunks, device=batch_edges.device).view([self.num_chunks, -1])
                mask = inv_mask == 0
                tmp_idx = torch.nonzero(mask)
                id_offset = deg_neg_src.flatten(0, 1).index_select(0, (tmp_idx.select(1, 0) * num_deg + tmp_idx.select(1,1)))
                sample_offset = tmp_idx.select(1, 1)
                src_neg_filter = id_offset * self.num_negs + (num_uni + sample_offset)

                deg_neg_dst = torch.randint(batch_edges.size(1), size=[self.num_chunks, num_deg], device=batch_edges.device)
                chunk_ids = deg_neg_dst.div(num_per_chunk).view([self.num_chunks, -1])
                inv_mask = chunk_ids - torch.arange(0, self.num_chunks, device=batch_edges.device).view([self.num_chunks, -1])
                mask = inv_mask == 0
                tmp_idx = torch.nonzero(mask)
                id_offset = deg_neg_dst.flatten(0, 1).index_select(0, (tmp_idx.select(1, 0) * num_deg + tmp_idx.select(1,1)))
                sample_offset = tmp_idx.select(1, 1)
                dst_neg_filter = id_offset * self.num_negs + (num_uni + sample_offset)

                neg_src = torch.cat([uni_neg_src, batch_edges[0, deg_neg_src]], 1)
                neg_dst = torch.cat([uni_neg_dst, batch_edges[-1, deg_neg_dst]], 1)
            else:
                neg_src = uni_neg_src
                neg_dst = uni_neg_dst

                src_neg_filter = None
                dst_neg_filter = None
        else:

            # assuming using all negatives
            neg_src = torch.arange(self.data.num_nodes, device=batch_edges.device)
            neg_dst = torch.arange(self.data.num_nodes, device=batch_edges.device)

            src = batch_edges[0]
            dst = batch_edges[-1]
            if batch_edges.shape[0] == 3:
                rel = batch_edges[1]
            else:
                rel = 0

            partial_src_rel_id = src + (rel * (self.data.num_nodes ** 2))
            partial_dst_rel_id = (dst * self.data.num_nodes) + (rel * (self.data.num_nodes ** 2))

            src_neg_edge_ids = partial_dst_rel_id.unsqueeze(1).expand(-1, self.data.num_nodes) + \
                               neg_src.unsqueeze(0)
            dst_neg_edge_ids = partial_src_rel_id.unsqueeze(1).expand(-1, self.data.num_nodes) + \
                               (neg_dst * self.data.num_nodes).unsqueeze(0)

            src_neg_idx = torch.bucketize(src_neg_edge_ids.flatten(0, 1), self.sorted_edge_ids)
            dst_neg_idx = torch.bucketize(dst_neg_edge_ids.flatten(0, 1), self.sorted_edge_ids)

            src_neg_filter = (self.sorted_edge_ids[src_neg_idx] == src_neg_edge_ids.flatten(0, 1)).nonzero().flatten(0,
                                                                                                                     1)
            dst_neg_filter = (self.sorted_edge_ids[dst_neg_idx] == dst_neg_edge_ids.flatten(0, 1)).nonzero().flatten(0,
                                                                                                                     1)

            neg_src = neg_src.unsqueeze(0)
            neg_dst = neg_dst.unsqueeze(0)

        return neg_src, neg_dst, src_neg_filter, dst_neg_filter
