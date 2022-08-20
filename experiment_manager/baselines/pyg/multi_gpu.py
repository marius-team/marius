import os
import time
import torch
import torch.nn.functional as F
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from tqdm import tqdm

import torch_geometric.loader.neighbor_loader as pyg_loader
from torch_geometric.nn import SAGEConv
from datasets import get_marius_dataset_nc


class SAGE(torch.nn.Module):

    def __init__(self, c_args):
        super().__init__()

        self.c_args = c_args
        dims = self.c_args.dims
        self.convs = torch.nn.ModuleList()

        for i in range(len(dims) - 1):
            input_dim = dims[i]
            output_dim = dims[i + 1]
            self.convs.append(SAGEConv(input_dim, output_dim, root_weight=True))

    def forward(self, x, nbrs, num_root):
        for i, conv in enumerate(self.convs):
            x = conv(x, nbrs)
            if i < len(self.convs) - 1:
                x = torch.relu(x)

        return x[:num_root]


def run(rank, world_size, data, c_args, nbr_sampler):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group('nccl', rank=rank, world_size=world_size)

    model = SAGE(c_args).to(rank)
    model = DistributedDataParallel(model, device_ids=[rank])
    optimizer = torch.optim.Adam(model.parameters(), lr=c_args.encoder_lr)

    train_idx = data.train_nodes.split(data.train_nodes.size(0) // world_size)[rank]



    print("Creating train loader")
    t0 = time.time()
    train_loader = pyg_loader.NeighborLoader(data,
                                             batch_size=c_args.training_batch_size,
                                             input_nodes=train_idx,
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

    for epoch in range(c_args.num_epochs):
        model.train()

        for batch in tqdm(train_loader):
            batch = batch.to(rank, non_blocking=True)
            model.zero_grad()
            out = model.forward(batch.x, batch.edge_index, batch.batch_size)
            loss = F.cross_entropy(out, batch.y[:batch.batch_size].to(int))
            loss.backward()
            optimizer.step()

        dist.barrier()

        if rank == 0:
            print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}')

        if rank == 0 and epoch % 1 == 0:  # We evaluate on a single GPU for now
            model.eval()
            with torch.no_grad():
                all_y_true = []
                all_y_pred = []

                for batch in tqdm(valid_loader):
                    batch = batch.to(rank)

                    out = model.forward(batch.x, batch.edge_index, batch.batch_size)

                    y_true = batch.y[:batch.batch_size].to(int)
                    y_pred = out.argmax(dim=-1)

                    all_y_true.append(y_true)
                    all_y_pred.append(y_pred)

                all_y_true = torch.cat(all_y_true)
                all_y_pred = torch.cat(all_y_pred)

                valid_acc = (all_y_true == all_y_pred).nonzero().size(0) / all_y_true.size(0)

                all_y_true = []
                all_y_pred = []

                for batch in tqdm(test_loader):
                    batch = batch.to(rank)

                    out = model.forward(batch.x, batch.edge_index, batch.batch_size)

                    y_true = batch.y[:batch.batch_size].to(int)
                    y_pred = out.argmax(dim=-1)

                    all_y_true.append(y_true)
                    all_y_pred.append(y_pred)

                all_y_true = torch.cat(all_y_true)
                all_y_pred = torch.cat(all_y_pred)

                test_acc = (all_y_true == all_y_pred).nonzero().size(0) / all_y_true.size(0)

                print("VALID/TEST Acc: {:.4f}/{:.4f}".format(valid_acc, test_acc))

        dist.barrier()

    dist.destroy_process_group()


def run_multi_gpu_nc(c_args):
    data = get_marius_dataset_nc(c_args.dataset, c_args.add_reverse_edges).share_memory_()

    print("Creating nbr sampler")
    t0 = time.time()
    nbr_sampler = pyg_loader.NeighborSampler(data, c_args.neighbors,
                                             replace=True,
                                             directed=True)
    t1 = time.time()
    print("Sampler creation took {:.4f}s".format(t1 - t0))

    world_size = torch.cuda.device_count()
    mp.spawn(run, args=(world_size, data, c_args, nbr_sampler), nprocs=world_size, join=True)
