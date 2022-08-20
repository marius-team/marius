import dgl
import torch
import timeit
import time
import dgl.multiprocessing as mp

from datasets import get_marius_dataset_nc
from models import GraphSage, GAT

CPU = torch.device("cpu")
GPU = torch.device("cuda")

MAX_EDGES = 2 * 1E9


def train_trace_one_epoch(encoder, loss_fxn, model_optimizer, data_loader, epoch_num, num_batches, proc_id, no_compute=False):
    start_time = timeit.default_timer()

    encoder.train()

    print_frequency = num_batches // 20
    batch_num = 0

    t1 = time.time()
    for input_nodes, output_nodes, blocks in data_loader:
        if batch_num % print_frequency == 0 and proc_id == 0:
            print("Time: {:.3f}, batch: {}/{}".format(timeit.default_timer(), batch_num, num_batches))
        batch_num += 1

        t2 = time.time()

        num_edges = 0
        for b in blocks:
            num_edges += b.num_edges()
        num_nodes = blocks[0].number_of_src_nodes()

        if not no_compute:
            input_nodes = input_nodes.to(GPU)
            output_nodes = output_nodes.to(GPU)
            blocks = [b.to(GPU) for b in blocks]

            torch.cuda.synchronize()
            t3 = time.time()

            predictions = encoder(blocks, blocks[0].srcdata['feat'])
            labels = blocks[-1].dstdata['label']
            loss = loss_fxn(predictions, labels.to(torch.int64))

            model_optimizer.zero_grad()
            loss.backward()
            model_optimizer.step()
            torch.cuda.synchronize()
            t4 = time.time()

            print(
                "LOAD {:.4f} TRANSFER {:.4f} COMPUTE {:.4f} NODES {} EDGES {}".format(t2 - t1,
                                                                                      t3 - t2,
                                                                                      t4 - t3,
                                                                                      num_nodes,
                                                                                      num_edges))
        else:
            print(
                "LOAD {:.4f} TRANSFER {:.4f} COMPUTE {:.4f} NODES {} EDGES {}".format(t2 - t1,
                                                                                      -1,
                                                                                      -1,
                                                                                      num_nodes,
                                                                                      num_edges))

        t1 = time.time()

    if proc_id == 0:
        print("Completed Epoch: {}, training time: {:.3f}s".format(epoch_num, timeit.default_timer() - start_time))


def train_one_epoch(encoder, loss_fxn, model_optimizer, data_loader, epoch_num, num_batches, proc_id):
    start_time = timeit.default_timer()

    encoder.train()

    print_frequency = num_batches // 20
    batch_num = 0
    for input_nodes, output_nodes, blocks in data_loader:
        if batch_num % print_frequency == 0 and proc_id == 0:
            print("Time: {:.3f}, batch: {}/{}".format(timeit.default_timer(), batch_num, num_batches))
        batch_num += 1

        predictions = encoder(blocks, blocks[0].srcdata['feat'])
        labels = blocks[-1].dstdata['label']
        loss = loss_fxn(predictions, labels.to(torch.int64))

        model_optimizer.zero_grad()
        loss.backward()
        model_optimizer.step()

    if proc_id == 0:
        print("Completed Epoch: {}, training time: {:.3f}s".format(epoch_num, timeit.default_timer() - start_time))


def eval_one_epoch(encoder, data_loader, split):
    start_time = timeit.default_timer()
    y_true = []
    y_pred = []

    encoder.eval()

    with torch.no_grad():
        for input_nodes, output_nodes, blocks in data_loader:
            predictions = encoder(blocks, blocks[0].srcdata['feat'])
            labels = blocks[-1].dstdata['label']

            # calculate accuracy
            y_true.append(labels)
            y_pred.append(predictions.argmax(1))

        y_true = torch.cat(y_true)
        y_pred = torch.cat(y_pred)
        accuracy = 100 * (y_true == y_pred).nonzero().shape[0] / y_true.shape[0]

        print("Accuracy for {} split: {:.4f}, evaluation time: {:.3f}s".format(split, accuracy,
                                                                               timeit.default_timer() - start_time))


def run(proc_id, devices, num_gpus, data, all_args):
    device_id = devices[proc_id]
    if num_gpus > 1:
        dist_init_method = 'tcp://{master_ip}:{master_port}'.format(master_ip='127.0.0.1', master_port='12345')
        torch.cuda.set_device(device_id)
        device = torch.device('cuda:' + str(device_id))
        torch.distributed.init_process_group(backend="nccl", init_method=dist_init_method, world_size=num_gpus,
                                             rank=proc_id)
    elif num_gpus == 0:
        device = CPU
    else:
        device = GPU

    full_graph, train_node_ids, valid_node_ids, test_node_ids = data

    sample_device, feat_dim, args = all_args

    # train data loaders
    train_nbr_sampler = dgl.dataloading.MultiLayerNeighborSampler(args.num_train_nbrs, replace=True)
    if sample_device != CPU:
        full_graph = full_graph.to(device)
        train_node_ids = train_node_ids.to(device)
        kwargs = {'batch_size': args.train_batch_size, 'shuffle': True, 'drop_last': False, 'pin_memory': False,
                  'num_workers': 0}
    else:
        kwargs = {'batch_size': args.train_batch_size, 'shuffle': True, 'drop_last': False, 'pin_memory': False,
                  'num_workers': args.num_workers, 'prefetch_factor': args.prefetch_factor,
                  'persistent_workers': args.persistent_workers}

        if args.num_workers == 0:
            kwargs.pop('prefetch_factor')
            kwargs.pop('persistent_workers')

    if args.print_timing:
        train_dl = dgl.dataloading.NodeDataLoader(full_graph,
                                                  train_node_ids,
                                                  train_nbr_sampler,
                                                  device=CPU,
                                                  use_ddp=num_gpus > 1,
                                                  **kwargs)
    else:
        train_dl = dgl.dataloading.NodeDataLoader(full_graph, train_node_ids, train_nbr_sampler, device=device,
                                                  use_ddp=num_gpus > 1, **kwargs)

    # eval data loaders
    eval_nbr_sampler = dgl.dataloading.MultiLayerNeighborSampler(args.num_eval_nbrs, replace=True)
    if sample_device != CPU:
        valid_node_ids = valid_node_ids.to(device)
        test_node_ids = test_node_ids.to(device)
        kwargs = {'batch_size': args.eval_batch_size, 'shuffle': True, 'drop_last': False, 'pin_memory': False,
                  'num_workers': 0}
    else:
        kwargs = {'batch_size': args.eval_batch_size, 'shuffle': True, 'drop_last': False, 'pin_memory': False,
                  'num_workers': args.num_workers, 'prefetch_factor': args.prefetch_factor,
                  'persistent_workers': args.persistent_workers}
        if args.num_workers == 0:
            kwargs.pop('prefetch_factor')
            kwargs.pop('persistent_workers')
    valid_dl = dgl.dataloading.NodeDataLoader(full_graph, valid_node_ids, eval_nbr_sampler, device=device, **kwargs)
    test_dl = dgl.dataloading.NodeDataLoader(full_graph, test_node_ids, eval_nbr_sampler, device=device, **kwargs)

    # models
    if args.model == 'graph_sage':
        encoder = GraphSage(feat_dim, args.h_dim, args.out_dim, args.num_layers,
                            aggregator_type=args.graph_sage_aggregator,
                            feat_drop=args.graph_sage_dropout, bias=True, norm=None)
    elif args.model == 'gat':
        encoder = GAT(feat_dim, args.h_dim, args.out_dim, args.gat_num_heads, args.num_layers,
                      feat_drop=args.gat_feat_drop, attn_drop=args.gat_attn_drop,
                      negative_slope=args.gat_negative_slope, bias=True, allow_zero=True)
    else:
        raise Exception("Invalid Encoder")

    encoder = encoder.to(device)
    if num_gpus > 1:
        encoder = torch.nn.parallel.DistributedDataParallel(encoder, device_ids=[device], output_device=device)
    loss_fxn = torch.nn.CrossEntropyLoss()

    if args.optimizer == 'Adam':
        model_optimizer = torch.optim.Adam(encoder.parameters(), lr=args.learning_rate)
    elif args.optimizer == 'Adagrad':
        model_optimizer = torch.optim.Adagrad(encoder.parameters(), lr=args.learning_rate)
    else:
        raise Exception("Invalid Optimizer")

    # training
    for epoch in range(1, args.num_epochs + 1):
        if proc_id == 0:
            print("\nStarting epoch {}".format(epoch))

        if num_gpus > 1:
            train_dl.set_epoch(epoch - 1)

        if all_args[2].print_timing:
            train_trace_one_epoch(encoder, loss_fxn, model_optimizer, train_dl, epoch,
                                  (train_node_ids.shape[0] // args.train_batch_size + 1) // num_gpus + 1, proc_id, no_compute=all_args[2].no_compute)
        else:
            train_one_epoch(encoder, loss_fxn, model_optimizer, train_dl, epoch,
                            (train_node_ids.shape[0] // args.train_batch_size + 1) // num_gpus + 1, proc_id)

        if proc_id == 0:
            eval_one_epoch(encoder, valid_dl, 'valid')
            eval_one_epoch(encoder, test_dl, 'test')

        if num_gpus > 1:
            torch.distributed.barrier()


def run_nc(args):
    if args.sample_device == "CPU":
        sample_device = CPU
    else:
        sample_device = GPU

    # dataset
    full_graph = get_marius_dataset_nc(args.base_directory, add_reverse_edges=False, only_sample=args.only_sample)
    num_nodes = full_graph.num_nodes()
    num_rels = torch.max(full_graph.edata['etype']).numpy() + 1 if 'etype' in full_graph.edge_attr_schemes() else 1

    if not args.only_sample:
        feat_dim = full_graph.ndata['feat'].shape[-1]
    else:
        feat_dim = 128

    print("Full Graph:", full_graph)
    print("Num Rels: ", num_rels)
    print("Feat dim: ", feat_dim)

    # add reverse edges if needed for sampling outgoing nbrs
    if args.outgoing_nbrs:
        full_graph = dgl.add_reverse_edges(full_graph, copy_ndata=True, copy_edata=True)
        full_graph.edata.pop('_ID')
    else:
        pass

    if full_graph.num_edges() > MAX_EDGES:
        print("Converting full graph to long datatype.")
        full_graph = full_graph.long()

    if args.single_format:
        print("Using single format to save memory.")
        full_graph = full_graph.formats('csc')
    else:
        full_graph.create_formats_()

    # data
    all_node_ids = torch.arange(num_nodes, dtype=torch.int32, device=CPU)

    train_mask = full_graph.ndata['train_mask'].to(torch.bool)
    train_node_ids = all_node_ids.masked_select(train_mask)

    valid_mask = full_graph.ndata['val_mask'].to(torch.bool)
    valid_node_ids = all_node_ids.masked_select(valid_mask)

    test_mask = full_graph.ndata['test_mask'].to(torch.bool)
    test_node_ids = all_node_ids.masked_select(test_mask)

    if full_graph.num_edges() > MAX_EDGES:
        train_node_ids = train_node_ids.to(torch.int64)
        valid_node_ids = valid_node_ids.to(torch.int64)
        test_node_ids = test_node_ids.to(torch.int64)

    data = full_graph, train_node_ids, valid_node_ids, test_node_ids
    all_args = sample_device, feat_dim, args

    # run training
    devices = list(range(args.num_gpus))
    if args.num_gpus == 1:
        run(0, devices, args.num_gpus, data, all_args)
    else:
        procs = []
        for proc_id in range(args.num_gpus):
            p = mp.Process(target=run, args=(proc_id, devices, args.num_gpus, data, all_args))
            p.start()
            procs.append(p)
        for p in procs:
            p.join()

# if __name__ == "__main__":
#     main()
