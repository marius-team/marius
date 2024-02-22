import marius as m


def infer_lp(
    model: m.nn.Model,
    graph_storage: m.storage.GraphModelStorage,
    output_dir: str,
    metrics: list = None,
    save_scores: bool = False,
    save_ranks: bool = False,
    batch_size: int = 10000,
    num_nbrs: list = None,
    num_negs: int = None,
    num_chunks: int = 1,
    deg_frac: float = 0.0,
    filtered: bool = True,
):
    reporter = m.report.LinkPredictionReporter()

    for metric in metrics:
        reporter.add_metric(metric)

    neg_sampler = None
    if num_negs is None:
        for metric in metrics:
            if isinstance(metric, m.report.RankingMetric):
                raise RuntimeError("Ranking metrics require the negative sampling configuration to be provided.")

        # Set the decoder to only compute scores for the positives
        model.decoder.mode = m.config.EdgeDecoderMethod.ONLY_POS
    else:
        model.decoder.mode = m.config.EdgeDecoderMethod.CORRUPT_NODE
        neg_sampler = m.samplers.CorruptNodeNegativeSampler(num_chunks, num_negs, deg_frac, filtered)

    nbr_sampler = None
    if num_nbrs is not None:
        nbr_sampler = m.samplers.LayeredNeighborSampler(graph_storage, num_nbrs)
    # if not graph_storage.has_encoded() and num_nbrs is not None:
    #     nbr_sampler = m.samplers.LayeredNeighborSampler(graph_storage, num_nbrs)

    dataloader = m.data.DataLoader(
        graph_storage=graph_storage,
        neg_sampler=neg_sampler,
        nbr_sampler=nbr_sampler,
        batch_size=batch_size,
        learning_task="lp",
    )

    dataloader.initializeBatches()

    while dataloader.hasNextBatch():
        batch = dataloader.getBatch(model.device)

        pos, neg, inv_pos, inv_neg = model.forward_lp(batch, train=False)

        # if graph_storage.has_encoded():
        #     # batch.node_embeddings contains saved encoder outputs
        #     pos, neg, inv_pos, inv_neg = model.decoder.forward(batch.edges, batch.node_embeddings)
        # else:
        #     pos, neg, inv_pos, inv_neg = model.forward_lp(batch, train=False)

        reporter.add_result(pos, neg, batch.edges)
        if inv_pos is not None:
            reporter.add_result(inv_pos, inv_neg, batch.edges)

        batch.clear()
        dataloader.finishedBatch()

    reporter.save(output_dir, save_scores, save_ranks)
