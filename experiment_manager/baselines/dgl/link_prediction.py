import dgl
import torch
import itertools
import timeit
import dgl.multiprocessing as mp

from datasets import get_marius_dataset_lp
from models import GraphEmbeddingLayer, DistMult, UniformChunkNegativeSampler, NegativeSamplerForFilteredMrr, \
    GraphSage, GAT

CPU = torch.device("cpu")
GPU = torch.device("cuda")



def filter_negatives(full_graph, rhs_neg_score, lhs_neg_score, negative_graph, rhs_neg_graph, lhs_neg_graph, num_nodes,
                     num_rels, device):
    all_src, all_dst, all_eid = full_graph.edges(form='all')
    if num_rels > 1:
        all_rels = full_graph.edata['etype'][all_eid.to(torch.int64)]
    else:
        all_rels = torch.zeros_like(all_src)

    all_src = all_src.to(device)
    all_dst = all_dst.to(device)
    all_rels = all_rels.to(device)

    # filter neg_score
    # all_src and all_dst are node ids from the original graph, need to convert them to the node ids of the
    # negative graph which are relabeled
    mapping = torch.argsort(negative_graph.ndata[dgl.NID], 0).to(torch.int32)
    local_all_src, local_all_dst, local_all_rels = mapping[all_src.to(torch.int64)], mapping[all_dst.to(torch.int64)], all_rels

    # filter rhs neg score
    rhs_neg_src_ids, rhs_neg_dst_ids, rhs_neg_eid = rhs_neg_graph.edges('all')
    rhs_neg_src_ids = rhs_neg_src_ids * num_rels + rhs_neg_graph.edata['etype'][rhs_neg_eid.to(torch.int64)]  # same as neg_etype
    rhs_neg_src_rel_graph = dgl.graph((rhs_neg_src_ids, rhs_neg_dst_ids), num_nodes=num_nodes * num_rels, device=device)

    rhs_local_all_src = local_all_src * num_rels + local_all_rels
    rhs_mask = rhs_neg_src_rel_graph.has_edges_between(rhs_local_all_src, local_all_dst)
    _, _, pos_edge_ids = rhs_neg_src_rel_graph.edge_ids(rhs_local_all_src[rhs_mask], local_all_dst[rhs_mask],
                                                        return_uv=True)
    rhs_neg_score[pos_edge_ids.to(torch.int64)] = -1e9

    # filter lhs neg_score
    lhs_neg_src_ids, lhs_neg_dst_ids, lhs_neg_eid = lhs_neg_graph.edges('all')
    lhs_neg_dst_ids = lhs_neg_dst_ids * num_rels + lhs_neg_graph.edata['etype'][lhs_neg_eid.to(torch.int64)]  # same as neg_etype
    lhs_neg_dst_rel_graph = dgl.graph((lhs_neg_src_ids, lhs_neg_dst_ids), num_nodes=num_nodes * num_rels, device=device)

    lhs_local_all_dst = local_all_dst * num_rels + local_all_rels
    lhs_mask = lhs_neg_dst_rel_graph.has_edges_between(local_all_src, lhs_local_all_dst)
    _, _, pos_edge_ids = lhs_neg_dst_rel_graph.edge_ids(local_all_src[lhs_mask], lhs_local_all_dst[lhs_mask],
                                                        return_uv=True)
    lhs_neg_score[pos_edge_ids.to(torch.int64)] = -1e9

    return rhs_neg_score, lhs_neg_score



def filter_deg_negatives(neg_scores, sample, edges_per_chunk, num_chunks, num_uniform_neg, num_deg_neg, device):
    sample = sample.view((num_chunks, num_deg_neg))

    chunk_ids = (sample.div(edges_per_chunk, rounding_mode="trunc")).view((num_chunks, -1))
    inv_mask = chunk_ids - torch.arange(0, num_chunks, device=device).view((num_chunks, -1))
    mask = (inv_mask == 0)
    temp_idx = torch.nonzero(mask)
    id_offset = sample.flatten(0, 1).index_select(0, (temp_idx.select(1, 0) * num_deg_neg + temp_idx.select(1, 1)))
    sample_offset = temp_idx.select(1, 1)
    sample_filter = id_offset * (num_uniform_neg + num_deg_neg) + (num_uniform_neg + sample_offset)

    neg_scores = neg_scores.flatten(0, 1).index_fill(0, sample_filter, -1e9).view((-1, num_uniform_neg + num_deg_neg))
    return neg_scores



def forward(input_nodes, blocks, positive_graph, negative_graph, embedding_layer, encoder, encode, decoder, num_chunks,
            num_uniform_negs, num_deg_negs, filtered_mrr, full_graph, num_nodes, num_rels, device):

    # should already be on GPU, but just in case
    # input_nodes = input_nodes.to(GPU)
    # blocks = [b.to(GPU) for b in blocks]
    # positive_graph = positive_graph.to(GPU)
    # negative_graph = negative_graph.to(GPU)

    if encode is True:
        input_features = embedding_layer(input_nodes, device)
        encoded_embs = encoder(blocks, input_features)
    else:
        input_features = embedding_layer(blocks[-1].dstdata[dgl.NID], device)
        encoded_embs = input_features

    rhs_pos_score = decoder(positive_graph, encoded_embs, direction='forward')
    lhs_pos_score = decoder(positive_graph, encoded_embs, direction='reverse')


    # negatives
    if filtered_mrr:
        num_uniform_negs = num_nodes

    if num_rels > 1:
        # negative graph doesn't retain edge type information
        neg_etype = torch.repeat_interleave(positive_graph.edata['etype'], num_uniform_negs, 0)
        neg_etype = neg_etype.repeat(2)
        negative_graph.edata['etype'] = neg_etype

    rhs_neg_graph = negative_graph.edge_subgraph(torch.arange(negative_graph.num_edges()//2,
                                                              dtype=torch.int32, device=device),
                                                 relabel_nodes=False, store_ids=False)
    rhs_neg_score = decoder(rhs_neg_graph, encoded_embs, direction='forward')

    lhs_neg_graph = negative_graph.edge_subgraph(torch.arange(negative_graph.num_edges()//2,
                                                              negative_graph.num_edges(),
                                                              dtype=torch.int32, device=device),
                                                 relabel_nodes=False, store_ids=False)
    lhs_neg_score = decoder(lhs_neg_graph, encoded_embs, direction='reverse')

    if filtered_mrr:
        rhs_neg_score, lhs_neg_score = filter_negatives(full_graph, rhs_neg_score, lhs_neg_score, negative_graph,
                                                        rhs_neg_graph, lhs_neg_graph, num_nodes, num_rels, device)

    rhs_neg_score = torch.reshape(rhs_neg_score, (rhs_pos_score.shape[0], num_uniform_negs))
    lhs_neg_score = torch.reshape(lhs_neg_score, (lhs_pos_score.shape[0], num_uniform_negs))


    # degree negatives
    if num_deg_negs > 0 and not filtered_mrr:
        batch_src, batch_dst = positive_graph.edges(form='uv')
        batch_size = batch_src.size(0)
        edges_per_chunk = torch.ceil(batch_size / num_chunks).to(torch.int32)

        if num_rels > 1:
            deg_neg_etype = torch.repeat_interleave(positive_graph.edata['etype'], num_deg_negs, 0)
        else:
            deg_neg_etype = None

        dst_sample = torch.randint(0, batch_size, (num_chunks * num_deg_negs,), device=device, dtype=batch_src.dtype)
        src = batch_src.repeat_interleave(num_deg_negs, 0)
        dst_neg = batch_dst.index_select(0, dst_sample).view((num_chunks, num_deg_negs)).repeat_interleave(edges_per_chunk, 0)
        dst_neg = dst_neg.reshape((edges_per_chunk * num_chunks * num_deg_negs,))
        dst_neg = dst_neg[:src.shape[0]]  # truncate if the last chunk doesn't have a full edges_per_chunk

        rhs_deg_neg_graph = dgl.graph((src, dst_neg), num_nodes=positive_graph.num_nodes(), device=device)
        if num_rels > 1:
            rhs_deg_neg_graph.edata['etype'] = deg_neg_etype

        src_sample = torch.randint(0, batch_size, (num_chunks * num_deg_negs,), device=device, dtype=batch_src.dtype)
        dst = batch_dst.repeat_interleave(num_deg_negs, 0)
        src_neg = batch_src.index_select(0, src_sample).view((num_chunks, num_deg_negs)).repeat_interleave(edges_per_chunk, 0)
        src_neg = src_neg.reshape((edges_per_chunk * num_chunks * num_deg_negs,))
        src_neg = src_neg[:dst.shape[0]]  # truncate if the last chunk doesn't have a full edges_per_chunk

        lhs_deg_neg_graph = dgl.graph((src_neg, dst), num_nodes=positive_graph.num_nodes(), device=device)
        if num_rels > 1:
            lhs_deg_neg_graph.edata['etype'] = deg_neg_etype

        rhs_deg_neg_score = decoder(rhs_deg_neg_graph, encoded_embs, direction='forward')
        lhs_deg_neg_score = decoder(lhs_deg_neg_graph, encoded_embs, direction='reverse')

        rhs_deg_neg_score = torch.reshape(rhs_deg_neg_score, (rhs_pos_score.shape[0], num_deg_negs))
        lhs_deg_neg_score = torch.reshape(lhs_deg_neg_score, (lhs_pos_score.shape[0], num_deg_negs))

        rhs_neg_score = torch.cat([rhs_neg_score, rhs_deg_neg_score], 1)
        lhs_neg_score = torch.cat([lhs_neg_score, lhs_deg_neg_score], 1)
        rhs_neg_score = filter_deg_negatives(rhs_neg_score, dst_sample, edges_per_chunk, num_chunks, num_uniform_negs,
                                             num_deg_negs, device)
        lhs_neg_score = filter_deg_negatives(lhs_neg_score, src_sample, edges_per_chunk, num_chunks, num_uniform_negs,
                                             num_deg_negs, device)

    return rhs_pos_score, rhs_neg_score, lhs_pos_score, lhs_neg_score



def train_one_epoch(embedding_layer, encoder, decoder, loss_fxn, model_optimizer, emb_optimizer, data_loader, encode,
                    num_chunks, num_uniform_negs, num_deg_negs, epoch_num, num_batches, num_nodes, num_rels, device,
                    proc_id):
    num_chunks = torch.tensor(num_chunks, device=device)

    start_time = timeit.default_timer()

    embedding_layer.train()
    encoder.train()
    decoder.train()

    print_frequency = num_batches//20
    batch_num = 0
    for input_nodes, positive_graph, negative_graph, blocks in data_loader:
        if batch_num % print_frequency == 0 and proc_id == 0:
            print("Time: {:.3f}, batch: {}/{}".format(timeit.default_timer(), batch_num, num_batches))
        batch_num += 1

        rhs_pos_score, rhs_neg_score, lhs_pos_score, lhs_neg_score = \
            forward(input_nodes, blocks, positive_graph, negative_graph, embedding_layer, encoder, encode, decoder,
                    num_chunks, num_uniform_negs, num_deg_negs, False, None, num_nodes, num_rels, device)

        # calculate loss
        rhs_neg_score = torch.logsumexp(rhs_neg_score, 1)
        lhs_neg_score = torch.logsumexp(lhs_neg_score, 1)

        rhs_scores = torch.stack([rhs_pos_score, rhs_neg_score], 1)
        lhs_scores = torch.stack([lhs_pos_score, lhs_neg_score], 1)
        target_scores = torch.zeros(rhs_pos_score.shape[0], dtype=torch.long, device=device)

        loss = loss_fxn(rhs_scores, target_scores) + loss_fxn(lhs_scores, target_scores)

        model_optimizer.zero_grad()
        emb_optimizer.zero_grad()
        loss.backward()
        model_optimizer.step()
        emb_optimizer.step()

    if proc_id == 0:
        print("Completed Epoch: {}, training time: {:.3f}s".format(epoch_num, timeit.default_timer() - start_time))



def eval_one_epoch(embedding_layer, encoder, decoder, data_loader, encode, num_chunks, num_uniform_negs, num_deg_negs,
                   num_eval_edges, filtered_mrr, full_graph, num_nodes, num_rels, device, split):
    num_chunks = torch.tensor(num_chunks, device=device)

    start_time = timeit.default_timer()

    embedding_layer.eval()
    encoder.eval()
    decoder.eval()

    all_ranks = torch.empty(num_eval_edges, dtype=torch.float32, device=device)

    counter = 0
    with torch.no_grad():
        for input_nodes, positive_graph, negative_graph, blocks in data_loader:
            rhs_pos_score, rhs_neg_score, lhs_pos_score, lhs_neg_score = \
                forward(input_nodes, blocks, positive_graph, negative_graph, embedding_layer, encoder, encode, decoder,
                        num_chunks, num_uniform_negs, num_deg_negs, filtered_mrr, full_graph, num_nodes, num_rels, device)

            # calculate ranks
            rhs_ranks = torch.greater_equal(rhs_neg_score, torch.unsqueeze(rhs_pos_score, 1)).to(dtype=torch.int32)
            rhs_ranks = torch.sum(rhs_ranks, 1) + 1
            rhs_ranks = rhs_ranks.to(torch.float32)

            lhs_ranks = torch.greater_equal(lhs_neg_score, torch.unsqueeze(lhs_pos_score, 1)).to(dtype=torch.int32)
            lhs_ranks = torch.sum(lhs_ranks, 1) + 1
            lhs_ranks = lhs_ranks.to(torch.float32)

            all_ranks[counter:counter + 2 * rhs_pos_score.shape[0]] = torch.cat([rhs_ranks, lhs_ranks], 0)
            counter = counter + 2 * rhs_pos_score.shape[0]

        mrr = torch.mean(torch.reciprocal(all_ranks))

        print("MRR for {} split: {:.4f}, evaluation time: {:.3f}s".format(split, mrr,
                                                                          timeit.default_timer() - start_time))
        for ii in [1, 3, 5, 10, 50, 100]:
            hits_at_k = all_ranks.le(ii).nonzero().size(0) / all_ranks.shape[0]
            print("Hits@{} for {} split: {:.4f}".format(ii, split, hits_at_k))



def run(proc_id, devices, num_gpus, data, all_args):
    device_id = devices[proc_id]
    if num_gpus > 1:
        dist_init_method = 'tcp://{master_ip}:{master_port}'.format(master_ip='127.0.0.1', master_port='12345')
        torch.cuda.set_device(device_id)
        device = torch.device('cuda:' + str(device_id))
        torch.distributed.init_process_group(backend="nccl", init_method=dist_init_method, world_size=num_gpus,
                                             rank=proc_id)
    else:
        device = GPU

    full_graph, train_g, valid_g, test_g, train_eids, valid_eids, test_eids, embedding_layer, emb_optimizer = data

    emb_storage_device, num_nodes, num_rels, args = all_args

    # train data loaders
    # nbr sampler seems to only get incoming neighbors, nbrs of all edge types
    # device is where to put the output of each iteration, currently train_g must be on CPU
    # gpu device seems to be better than cpu device with pin memory true (or false, f ~ equal or slightly faster than true)
    # (pin memory must be false if device is gpu)
    # num_workers does seem to reduce time, 4 ~= 16 on FB15k-237, prefetch_factor 2 faster than 8, neither seem to hurt accuracy
    # persistent_workers reduces setup time of subsequent epochs
    # several configs seem to give best performance, seems it is best not to overload workers/prefetch
    train_nbr_sampler = dgl.dataloading.MultiLayerNeighborSampler(args.num_train_nbrs, replace=True)
    train_neg_sampler = UniformChunkNegativeSampler(args.num_train_chunks, args.num_train_uniform_negs)
    kwargs = {'g_sampling': None, 'exclude': None, 'negative_sampler': train_neg_sampler,
              'batch_size': args.train_batch_size, 'shuffle': True, 'drop_last': False, 'pin_memory': False,
              'num_workers': args.num_workers, 'prefetch_factor': args.prefetch_factor,
              'persistent_workers': args.persistent_workers}
    if args.num_workers == 0:
        kwargs.pop('prefetch_factor')
        kwargs.pop('persistent_workers')
    train_dl = dgl.dataloading.EdgeDataLoader(train_g, train_eids, train_nbr_sampler, device=device,
                                              use_ddp=num_gpus > 1, **kwargs)

    eval_nbr_sampler = dgl.dataloading.MultiLayerNeighborSampler(args.num_eval_nbrs, replace=True)
    if args.filtered_mrr:
        eval_neg_sampler = NegativeSamplerForFilteredMrr()
    else:
        eval_neg_sampler = UniformChunkNegativeSampler(args.num_eval_chunks, args.num_eval_uniform_negs)
    kwargs = {'g_sampling': train_g, 'exclude': None, 'negative_sampler': eval_neg_sampler,
              'batch_size': args.eval_batch_size, 'shuffle': True, 'drop_last': False, 'pin_memory': False,
              'num_workers': args.num_workers, 'prefetch_factor': args.prefetch_factor,
              'persistent_workers': args.persistent_workers}
    if args.num_workers == 0:
        kwargs.pop('prefetch_factor')
        kwargs.pop('persistent_workers')
    eval_valid_dl = dgl.dataloading.EdgeDataLoader(valid_g, valid_eids, eval_nbr_sampler, device=device, **kwargs)
    eval_test_dl = dgl.dataloading.EdgeDataLoader(test_g, test_eids, eval_nbr_sampler, device=device, **kwargs)

    # models
    # embedding_layer = GraphEmbeddingLayer(num_nodes, emb_dim, storage_device=emb_storage_device,
    #                                       backend=emb_storage_backend)
    if args.model == 'graph_sage':
        encoder = GraphSage(args.emb_dim, args.h_dim, args.out_dim, args.num_layers,
                            aggregator_type=args.graph_sage_aggregator, feat_drop=args.graph_sage_dropout,
                            bias=True, norm=None)
    elif args.model == 'gat':
        encoder = GAT(args.emb_dim, args.h_dim, args.out_dim, args.gat_num_heads, args.num_layers,
                      feat_drop=args.gat_feat_drop, attn_drop=args.gat_attn_drop,
                      negative_slope=args.gat_negative_slope, bias=True, allow_zero=True)
    else:
        print("Invalid Encoder")
        raise Exception()
    decoder = DistMult(num_rels, args.out_dim)

    encoder = encoder.to(device)
    decoder = decoder.to(device)
    if num_gpus > 1:
        encoder = torch.nn.parallel.DistributedDataParallel(encoder, device_ids=[device], output_device=device)
        decoder = torch.nn.parallel.DistributedDataParallel(decoder, device_ids=[device], output_device=device)
    loss_fxn = torch.nn.CrossEntropyLoss(reduction='sum')

    all_params = itertools.chain(encoder.parameters(), decoder.parameters())
    model_optimizer = torch.optim.Adagrad(all_params, lr=args.learning_rate)
    # if emb_storage_backend == 'dgl_sparse':
    #     emb_optimizer = dgl.optim.SparseAdagrad([embedding_layer.node_embs], lr=learning_rate)
    # else:
    #     emb_optimizer = torch.optim.Adagrad(embedding_layer.parameters(), lr=learning_rate)

    # training
    for epoch in range(1, args.num_epochs + 1):
        if proc_id == 0:
            print("\nStarting epoch {}".format(epoch))

        if num_gpus > 1:
            train_dl.set_epoch(epoch - 1)

        train_one_epoch(embedding_layer, encoder, decoder, loss_fxn, model_optimizer, emb_optimizer, train_dl,
                        args.encode, args.num_train_chunks, args.num_train_uniform_negs, args.num_train_deg_negs,
                        epoch, (train_eids.shape[0] // args.train_batch_size + 1) // num_gpus + 1, num_nodes, num_rels,
                        device, proc_id)

        if proc_id == 0 and epoch % args.epochs_per_eval == 0:
            eval_one_epoch(embedding_layer, encoder, decoder, eval_valid_dl, args.encode, args.num_eval_chunks,
                           args.num_eval_uniform_negs, args.num_eval_deg_negs, valid_g.num_edges() * 2,
                           args.filtered_mrr, full_graph, num_nodes, num_rels, device, 'valid')
            eval_one_epoch(embedding_layer, encoder, decoder, eval_test_dl, args.encode, args.num_eval_chunks,
                           args.num_eval_uniform_negs, args.num_eval_deg_negs, test_g.num_edges() * 2,
                           args.filtered_mrr, full_graph, num_nodes, num_rels, device, 'test')

        if num_gpus > 1:
            torch.distributed.barrier()



def run_lp(args):
    if args.emb_storage_device == "CPU":
        emb_storage_device = CPU
    else:
        emb_storage_device = GPU

    # graphs are 'homogeneous', even if they have typed edges, edge type is stored as an edata field 'etype'
    full_graph = get_marius_dataset_lp(args.base_directory, add_reverse_edges=False)
    num_nodes = full_graph.num_nodes()
    num_rels =  torch.max(full_graph.edata['etype']).numpy() + 1 if 'etype' in full_graph.edge_attr_schemes() else 1
    print("Full Graph:", full_graph)
    print("Num Rels: ", num_rels)

    # clear some useless data from the graph
    if 'ntype' in full_graph.node_attr_schemes():
        full_graph.ndata.pop('ntype')
    if 'train_edge_mask' in full_graph.edge_attr_schemes():
        full_graph.edata.pop('train_edge_mask')
    if 'valid_edge_mask' in full_graph.edge_attr_schemes():
        full_graph.edata.pop('valid_edge_mask')
    if 'test_edge_mask' in full_graph.edge_attr_schemes():
        full_graph.edata.pop('test_edge_mask')

    # train/valid/test graphs, retain the etype info
    train_mask = full_graph.edata['train_mask'].to(torch.bool)
    train_g = full_graph.edge_subgraph(train_mask, relabel_nodes=False, store_ids=False)
    train_g.edata.pop('train_mask')
    train_g.edata.pop('val_mask')
    train_g.edata.pop('test_mask')
    train_eids = train_g.edges(form='eid')
    if args.outgoing_nbrs:
        train_g = dgl.add_reverse_edges(train_g, copy_ndata=False, copy_edata=True)
        train_g.edata.pop('_ID')
        # train_g = dgl.add_self_loop(train_g)
    else:
        pass
    print("Train Graph:", train_g)

    valid_mask = full_graph.edata['val_mask'].to(torch.bool)
    valid_g = full_graph.edge_subgraph(valid_mask, relabel_nodes=False, store_ids=False)
    valid_g.edata.pop('train_mask')
    valid_g.edata.pop('val_mask')
    valid_g.edata.pop('test_mask')
    valid_eids = valid_g.edges(form='eid')
    print("Valid Graph:", valid_g)

    test_mask = full_graph.edata['test_mask'].to(torch.bool)
    test_g = full_graph.edge_subgraph(test_mask, relabel_nodes=False, store_ids=False)
    test_g.edata.pop('train_mask')
    test_g.edata.pop('val_mask')
    test_g.edata.pop('test_mask')
    test_eids = test_g.edges(form='eid')
    print("Test Graph:", test_g)

    full_graph.create_formats_()
    train_g.create_formats_()
    valid_g.create_formats_()
    test_g.create_formats_()

    embedding_layer = GraphEmbeddingLayer(num_nodes, args.emb_dim, storage_device=emb_storage_device,
                                          backend=args.emb_storage_backend)
    if args.emb_storage_backend == 'dgl_sparse':
        emb_optimizer = dgl.optim.SparseAdagrad([embedding_layer.node_embs], lr=args.learning_rate)
    else:
        emb_optimizer = torch.optim.Adagrad(embedding_layer.parameters(), lr=args.learning_rate)

    # data
    data = full_graph, train_g, valid_g, test_g, train_eids, valid_eids, test_eids, embedding_layer, emb_optimizer

    all_args = emb_storage_device, num_nodes, num_rels, args


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