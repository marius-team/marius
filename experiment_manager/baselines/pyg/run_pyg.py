import argparse
import math
import time
from tqdm import tqdm
import torch_geometric.loader.neighbor_loader as pyg_loader
from helpers import NeighborSampler, NegativeSampler, get_mapping
from models import compute_softmax, select_encoder, select_decoder, Model
from datasets import select_dataset
from multi_gpu import run_multi_gpu_nc
import torch

import torch.nn.functional as F


@torch.no_grad()
def eval_nc(eval_loader, model):
    compute_device = model.device

    all_y_true = []
    all_y_pred = []

    for batch in iter(eval_loader):
        batch = batch.to(compute_device)

        out = model.encoder(batch.x, batch.edge_index, batch.batch_size)

        y_true = batch.y[:batch.batch_size].to(int)
        y_pred = out.argmax(dim=-1)

        all_y_true.append(y_true)
        all_y_pred.append(y_pred)

    all_y_true = torch.cat(all_y_true)
    all_y_pred = torch.cat(all_y_pred)

    acc = (all_y_true == all_y_pred).nonzero().size(0) / all_y_true.size(0)
    return acc


def train_nc(train_loader: pyg_loader.NeighborLoader, model, no_compute=False):

    compute_device = model.device

    t0 = time.time()
    for batch in iter(train_loader):

        num_nodes = batch.num_nodes
        num_edges = batch.edge_index.size(1)

        t1 = time.time()
        if not no_compute:
            batch = batch.to(compute_device, non_blocking=True)

            if config_args.print_timing:
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                t2 = time.time()

            model.zero_grad()
            out = model.encoder(batch.x, batch.edge_index, batch.batch_size)
            loss = F.cross_entropy(out, batch.y[:batch.batch_size].to(int))

            loss.backward()
            model.step()

            if config_args.print_timing:
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                t3 = time.time()

            if config_args.print_timing:
                print("LOAD {:.4f} TRANSFER {:.4f} COMPUTE {:.4f} NODES {} EDGES {}".format(t1 - t0,
                                                                                            t2 - t1,
                                                                                            t3 - t2,
                                                                                            num_nodes,
                                                                                            num_edges
                                                                                            ))
        else:
            print("LOAD {:.4f} TRANSFER {:.4f} COMPUTE {:.4f} NODES {} EDGES {}".format(t1 - t0,
                                                                                        -1,
                                                                                        -1,
                                                                                        num_nodes,
                                                                                        num_edges
                                                                                        ))

        t0 = time.time()


def run_iteration_lp(edges, edge_types, negative_sampler, neighbor_sampler, model, neighbors, train=True):
    compute_device = model.device

    t0 = time.time()
    neg_src, neg_dst, src_neg_filter, dst_neg_filter = negative_sampler.sample(edges)
    t1 = time.time()

    root_nodes, src_mapping, dst_mapping, src_neg_mapping, dst_neg_mapping = get_mapping(edges, neg_src, neg_dst)
    t2 = time.time()

    all_node_ids = root_nodes
    if neighbor_sampler is not None:
        all_node_ids, nbrs = neighbor_sampler.sample_nbrs(all_node_ids, num_neighbors=neighbors)
    t3 = time.time()

    src_mapping = src_mapping.to(compute_device)
    dst_mapping = dst_mapping.to(compute_device)
    src_neg_mapping = src_neg_mapping.to(compute_device)
    dst_neg_mapping = dst_neg_mapping.to(compute_device)
    edge_types = edge_types.to(compute_device)

    if src_neg_filter is not None:
        src_neg_filter = src_neg_filter.to(compute_device)
        dst_neg_filter = dst_neg_filter.to(compute_device)

    t4 = time.time()

    all_node_embs = model.node_embs[all_node_ids].to(compute_device)
    all_node_embs = all_node_embs.detach()

    if train:
        all_node_embs.requires_grad_()

    x = all_node_embs
    if neighbor_sampler is not None:
        x = model.encoder.forward(x, nbrs.edge_index.to(compute_device), root_nodes.size(0)).to(
            compute_device)

    if config_args.print_timing:
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t5 = time.time()

    src_emb = x[src_mapping]
    dst_emb = x[dst_mapping]
    src_neg_emb = x[src_neg_mapping].view([1, neg_src.flatten(0, 1).size(0), -1])
    dst_neg_emb = x[dst_neg_mapping].view([1, neg_dst.flatten(0, 1).size(0), -1])

    if config_args.print_timing:
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t6 = time.time()

    src_pos_score, src_neg_score = model.decoder.forward(src_emb, dst_emb, edge_types, src_neg_emb, corruption="src")
    dst_pos_score, dst_neg_score = model.decoder.forward(src_emb, dst_emb, edge_types, dst_neg_emb, corruption="dst")

    if src_neg_filter is not None:
        src_neg_score.flatten(0, 1)[src_neg_filter] = -1e9
        dst_neg_score.flatten(0, 1)[dst_neg_filter] = -1e9

    if config_args.print_timing:
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t7 = time.time()

    if config_args.print_timing:
        print(
            "NEGS {:.4f}s, UNIQUE {:.4f}s, NBRS {:.4f}s, TRANSFER {:.4f}s, ENCODER {:.4f}s, MAP {:.4f}s, DECODER {:.4f}s".format(
                t1 - t0, t2 - t1, t3 - t2, t4 - t3, t5 - t4, t6 - t5, t7 - t6))

    return src_pos_score, src_neg_score, dst_pos_score, dst_neg_score, all_node_ids, all_node_embs


@torch.no_grad()
def eval_lp(data, model, config_args, valid=True, has_nbrs=True, neighbors=[-1]):
    if valid:
        edge_index = data.valid_split
        edge_type = data.valid_edge_type
    else:
        edge_index = data.test_split
        edge_type = data.test_edge_type

    num_edges = edge_index.size(1)
    batch_size = config_args.eval_batch_size

    ranks = []

    num_batches = math.ceil(num_edges / batch_size)

    negative_sampler = NegativeSampler(data,
                                       1,
                                       config_args.eval_num_negs,
                                       config_args.eval_degree_fraction,
                                       config_args.eval_filtered)

    neighbor_sampler = None
    if has_nbrs:
        neighbor_sampler = NeighborSampler(data)

    edge_offset = 0
    size = batch_size

    for i in tqdm(range(num_batches), position=0, leave=True):

        if edge_offset + size > num_edges:
            size = num_edges - edge_offset

        edges = edge_index[:, edge_offset:edge_offset + size]

        if edge_type is not None:
            edge_types = edge_type[edge_offset:edge_offset + size]
            edges = torch.stack([edges[0], edge_types, edges[1]])
        else:
            edge_types = None
            edges = edges

        edge_offset += size

        src_pos_score, src_neg_score, dst_pos_score, dst_neg_score, _, _ = run_iteration_lp(edges,
                                                                                            edge_types,
                                                                                            negative_sampler,
                                                                                            neighbor_sampler,
                                                                                            model,
                                                                                            neighbors,
                                                                                            False)

        src_ranks = (src_neg_score >= src_pos_score.unsqueeze(1)).sum(1) + 1
        dst_ranks = (dst_neg_score >= dst_pos_score.unsqueeze(1)).sum(1) + 1

        ranks.append(src_ranks)
        ranks.append(dst_ranks)

    ranks = torch.cat(ranks)
    mrr = ranks.to(torch.float32).reciprocal().mean()

    return mrr, ranks


def train_lp(data, model, config_args, has_nbrs=True, neighbors=[-1]):
    edge_index = data.train_split
    edge_type = data.train_edge_type

    num_edges = edge_index.size(1)
    batch_size = config_args.training_batch_size

    num_batches = math.ceil(num_edges / batch_size)

    negative_sampler = NegativeSampler(data,
                                       config_args.training_num_chunks,
                                       config_args.training_num_negs,
                                       config_args.training_degree_fraction,
                                       False)

    neighbor_sampler = None
    if has_nbrs:
        neighbor_sampler = NeighborSampler(data)

    edge_offset = 0
    size = batch_size

    for i in tqdm(range(num_batches)):

        model.zero_grad()

        if edge_offset + size > num_edges:
            size = num_edges - edge_offset

        edges = edge_index[:, edge_offset:edge_offset + size]

        if edge_type is not None:
            edge_types = edge_type[edge_offset:edge_offset + size]
            edges = torch.stack([edges[0], edge_types, edges[1]])
        else:
            edge_types = None
            edges = edges

        edge_offset += size

        src_pos_score, src_neg_score, dst_pos_score, dst_neg_score, node_ids, all_node_embs = run_iteration_lp(edges,
                                                                                                               edge_types,
                                                                                                               negative_sampler,
                                                                                                               neighbor_sampler,
                                                                                                               model,
                                                                                                               neighbors)

        t0 = time.time()

        src_loss = compute_softmax(src_pos_score, src_neg_score)
        dst_loss = compute_softmax(dst_pos_score, dst_neg_score)
        loss = src_loss + dst_loss

        if config_args.print_timing:
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t1 = time.time()

        loss.backward()

        node_grads = all_node_embs.grad

        if config_args.print_timing:
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t2 = time.time()

        model.step(node_ids, node_grads)
        if config_args.print_timing:
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t3 = time.time()

        if config_args.print_timing:
            print("LOSS {:.4f}s, BACKWARD {:.4f}s, STEP {:.4f}s".format(
                t1 - t0, t2 - t1, t3 - t2))


def run_lp(c_args):
    edge_storage_device = torch.device("cpu")
    node_storage_device = torch.device("cpu")
    compute_device = torch.device("cpu")

    if c_args.node_storage.upper() == "GPU":
        assert torch.cuda.is_available()
        edge_storage_device = torch.device("cuda")

    if c_args.node_storage.upper() == "GPU":
        assert torch.cuda.is_available()
        node_storage_device = torch.device("cuda")

    if c_args.compute.upper() == "GPU":
        assert torch.cuda.is_available()
        compute_device = torch.device("cuda")

    data = select_dataset(c_args.dataset, "link_prediction", c_args.add_reverse_edges)
    data = data.to(edge_storage_device)

    c_args.__setattr__("num_nodes", data.num_nodes)
    c_args.__setattr__("num_relations", data.num_relations // 2)

    scale_factor = .001
    node_embs = torch.randn([c_args.num_nodes, c_args.dims[0]], device=node_storage_device) * scale_factor

    has_nbrs = True
    if c_args.model_encoder.upper() == "NONE":
        has_nbrs = False
        encoder = None
        encoder_optimizer = None
    else:
        encoder = select_encoder(c_args, compute_device)
        if c_args.encoder_optimizer.upper() == "ADAGRAD":
            encoder_optimizer = torch.optim.Adagrad(encoder.parameters(), lr=c_args.encoder_lr)
        elif c_args.encoder_optimizer.upper() == "ADAM":
            encoder_optimizer = torch.optim.Adam(encoder.parameters(), lr=c_args.encoder_lr)
        else:
            raise RuntimeError("Unrecognized optimizer")

    decoder = select_decoder(c_args, compute_device)
    if c_args.decoder_optimizer.upper() == "ADAGRAD":
        decoder_optimizer = torch.optim.Adagrad(decoder.parameters(), lr=c_args.decoder_lr)
    elif c_args.decoder_optimizer.upper() == "ADAM":
        decoder_optimizer = torch.optim.Adam(decoder.parameters(), lr=c_args.decoder_lr)
    else:
        raise RuntimeError("Unrecognized optimizer")

    model = Model(node_embs, encoder, decoder, encoder_optimizer, decoder_optimizer, compute_device)

    shuffle_ids = torch.randperm(data.valid_split.size(1), device=edge_storage_device)
    data.valid_split = data.valid_split.index_select(1, shuffle_ids)
    data.valid_edge_type = data.valid_edge_type.index_select(0, shuffle_ids)

    shuffle_ids = torch.randperm(data.test_split.size(1), device=edge_storage_device)
    data.test_split = data.test_split.index_select(1, shuffle_ids)
    data.test_edge_type = data.test_edge_type.index_select(0, shuffle_ids)

    for i in range(c_args.num_epochs):
        # shuffle edges on CPU
        shuffle_ids = torch.randperm(data.train_split.size(1), device=edge_storage_device)
        data.train_split = data.train_split.index_select(1, shuffle_ids)
        data.train_edge_type = data.train_edge_type.index_select(0, shuffle_ids)

        print("Running epoch: {}".format(i + 1))
        t0 = time.time()
        train_lp(data, model, c_args, has_nbrs=has_nbrs, neighbors=c_args.neighbors)
        t1 = time.time()
        print("Training took {:.4f}s".format(t1 - t0))

        valid_mrr, _ = eval_lp(data, model, c_args, True, has_nbrs=has_nbrs, neighbors=c_args.neighbors)
        test_mrr, _ = eval_lp(data, model, c_args, False, has_nbrs=has_nbrs, neighbors=c_args.neighbors)
        t2 = time.time()
        print("Testing took {:.4f}s".format(t2 - t1))

        print("VALID/TEST MRR: {:.4f}/{:.4f}".format(valid_mrr, test_mrr))


def run_nc(c_args):
    edge_storage_device = torch.device("cpu")
    compute_device = torch.device("cpu")

    if c_args.node_storage.upper() == "GPU":
        assert torch.cuda.is_available()
        edge_storage_device = torch.device("cuda")

    if c_args.node_storage.upper() == "GPU":
        assert torch.cuda.is_available()
        node_storage_device = torch.device("cuda")

    if c_args.compute.upper() == "GPU":
        assert torch.cuda.is_available()
        compute_device = torch.device("cuda")

    data = select_dataset(c_args.dataset, "node_classification", c_args.add_reverse_edges, c_args.only_sample).share_memory_()
    data = data.to(edge_storage_device)

    print("Creating nbr sampler")
    t0 = time.time()
    nbr_sampler = pyg_loader.NeighborSampler(data, c_args.neighbors, replace=True, directed=True)
    t1 = time.time()
    print("Sampler creation took {:.4f}s".format(t1 - t0))

    print("Creating train loader")
    t0 = time.time()
    train_loader = pyg_loader.NeighborLoader(data,
                                             batch_size=c_args.training_batch_size,
                                             input_nodes=data.train_nodes,
                                             num_neighbors=c_args.neighbors,
                                             pin_memory=True,
                                             num_workers=c_args.num_workers,
                                             shuffle=True,
                                             neighbor_sampler=nbr_sampler,
                                             replace=True,
                                             directed=True)
    t1 = time.time()
    print("Train loader creation took {:.4f}s".format(t1 - t0))

    print("Creating valid/test loaders")
    t0 = time.time()
    valid_loader = pyg_loader.NeighborLoader(data,
                                             batch_size=c_args.eval_batch_size,
                                             input_nodes=data.valid_nodes,
                                             num_neighbors=c_args.neighbors,
                                             neighbor_sampler=nbr_sampler,
                                             replace=True,
                                             directed=True)

    test_loader = pyg_loader.NeighborLoader(data,
                                            batch_size=c_args.eval_batch_size,
                                            input_nodes=data.test_nodes,
                                            num_neighbors=c_args.neighbors,
                                            neighbor_sampler=nbr_sampler,
                                            replace=True,
                                            directed=True)
    t1 = time.time()
    print("Valid/test loader creation took {:.4f}s".format(t1 - t0))

    c_args.__setattr__("num_nodes", data.num_nodes)
    c_args.__setattr__("num_relations", data.num_relations // 2)

    node_embs = None
    encoder = select_encoder(c_args, compute_device)

    if c_args.encoder_optimizer.upper() == "ADAGRAD":
        encoder_optimizer = torch.optim.Adagrad(encoder.parameters(), lr=c_args.encoder_lr)
    elif c_args.encoder_optimizer.upper() == "ADAM":
        encoder_optimizer = torch.optim.Adam(encoder.parameters(), lr=c_args.encoder_lr)
    else:
        raise RuntimeError("Unrecognized optimizer")
    decoder = None
    decoder_optimizer = None

    model = Model(node_embs, encoder, decoder, encoder_optimizer, decoder_optimizer, compute_device)

    for i in range(c_args.num_epochs):
        print("Running epoch: {}".format(i + 1))
        t0 = time.time()
        train_nc(train_loader, model, no_compute=c_args.no_compute)
        t1 = time.time()
        print("Training took {:.4f}s".format(t1 - t0))

        if not c_args.print_timing:
            valid_acc = eval_nc(valid_loader, model)
            test_acc = eval_nc(test_loader, model)
            t2 = time.time()
            print("Testing took {:.4f}s".format(t2 - t1))

            print("VALID/TEST Acc: {:.4f}/{:.4f}".format(valid_acc, test_acc))


if __name__ == "__main__":
    p = argparse.ArgumentParser()

    # [dataset_options]
    p.add_argument("--dataset")
    p.add_argument("--learning_task", default="link_prediction")
    p.add_argument("--model_encoder", default="GRAPH_SAGE")
    p.add_argument("--model_decoder", default="DistMult")
    p.add_argument("--model_loss", default="Softmax")
    p.add_argument('--add_reverse_edges', action='store_true', default=False)
    p.add_argument('--num_workers', type=int, default=1)

    # [embedding_options]
    p.add_argument("--dims", nargs="+", type=int, default=[50, 50])
    p.add_argument("--edge_storage", type=str, default="CPU")
    p.add_argument("--node_storage", type=str, default="CPU")
    p.add_argument("--compute", type=str, default="GPU")
    p.add_argument("--neighbors", nargs="+", type=int, default=[-1])

    # [training_options]
    p.add_argument("--training_batch_size", type=int, default=1000)
    p.add_argument("--training_num_chunks", type=int, default=10)
    p.add_argument("--training_num_negs", type=int, default=1000)
    p.add_argument("--training_degree_fraction", type=float, default=0.0)
    p.add_argument("--num_epochs", type=int, default=10)
    p.add_argument("--encoder_optimizer", type=str, default="Adagrad")
    p.add_argument("--encoder_lr", type=float, default=.1)
    p.add_argument("--decoder_optimizer", type=str, default="Adagrad")
    p.add_argument("--decoder_lr", type=float, default=.1)
    p.add_argument('--num_gpus', type=int, default=1)

    # [evaluation_options]
    p.add_argument('--eval_filtered', action='store_true', default=False)
    p.add_argument("--eval_batch_size", type=int, default=1000)
    p.add_argument("--eval_num_negs", type=int, default=1000)
    p.add_argument("--eval_degree_fraction", type=float, default=0.0)

    p.add_argument('--print_timing', action='store_true', default=False)
    p.add_argument('--only_sample', action='store_true', default=False)
    p.add_argument('--no_compute', action='store_true', default=False)

    config_args = p.parse_args()

    if config_args.num_gpus > 1:
        run_multi_gpu_nc(config_args)
    else:
        if config_args.learning_task.upper() == "LINK_PREDICTION":
            run_lp(config_args)
        else:
            run_nc(config_args)
