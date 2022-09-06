import marius as m


def infer_nc(
    model: m.nn.Model,
    graph_storage: m.storage.GraphModelStorage,
    output_dir: str,
    metrics: list = None,
    save_labels: bool = False,
    batch_size: int = 1000,
    num_nbrs: list = None,
):
    reporter = m.report.NodeClassificationReporter()

    for metric in metrics:
        reporter.add_metric(metric)

    nbr_sampler = None
    if num_nbrs is not None:
        nbr_sampler = m.samplers.LayeredNeighborSampler(graph_storage, num_nbrs)

    dataloader = m.data.DataLoader(
        graph_storage=graph_storage, nbr_sampler=nbr_sampler, batch_size=batch_size, learning_task="nc"
    )

    dataloader.initializeBatches()

    while dataloader.hasNextBatch():
        batch = dataloader.getBatch(model.device)
        labels = model.forward_nc(batch.node_embeddings, batch.node_features, batch.dense_graph, train=False)
        reporter.add_result(labels)
        batch.clear()
        dataloader.finishedBatch()

    reporter.save(output_dir, save_labels)
