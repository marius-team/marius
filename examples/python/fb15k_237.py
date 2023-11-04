from pathlib import Path

from omegaconf import OmegaConf

import marius as m
from marius.tools.preprocess.datasets.fb15k_237 import FB15K237

import torch  # isort:skip


def init_model(embedding_dim, num_nodes, num_relations, device, dtype):
    # setup shallow embedding encoder
    embedding_layer = m.nn.layers.EmbeddingLayer(dimension=embedding_dim, device=device)
    encoder = m.encoders.GeneralEncoder(layers=[[embedding_layer]])

    # initialize node embedding table
    emb_table = embedding_layer.init_embeddings(num_nodes)

    # initialize DistMult decoder
    decoder = m.nn.decoders.edge.DistMult(
        num_relations=num_relations,
        embedding_dim=embedding_dim,
        use_inverse_relations=True,
        device=device,
        dtype=dtype,
        mode="train",
    )

    loss = m.nn.SoftmaxCrossEntropy(reduction="sum")

    # metrics to compute during evaluation
    reporter = m.report.LinkPredictionReporter()
    reporter.add_metric(m.report.MeanReciprocalRank())
    reporter.add_metric(m.report.MeanRank())
    reporter.add_metric(m.report.Hitsk(1))
    reporter.add_metric(m.report.Hitsk(10))

    # sparse_lr sets the learning rate for the embedding parameters
    model = m.nn.Model(encoder, decoder, loss, reporter, sparse_lr=0.1)

    # set optimizer for dense model parameters. In this case this is the DistMult relation (edge-type) embeddings
    model.optimizers = [m.nn.AdamOptimizer(model.named_parameters(), lr=0.1)]

    return model, emb_table


def train_epoch(model, dataloader):
    # need to reset dataloader state before each epoch
    dataloader.initializeBatches()

    counter = 0
    while dataloader.hasNextBatch():
        batch = dataloader.getBatch()
        model.train_batch(batch)
        dataloader.updateEmbeddings(batch)

        counter += 1
        if counter % 100 == 0:
            print("Trained {} batches".format(counter))

    print("Trained {} batches".format(counter))


def eval_epoch(model, dataloader):
    # need to reset dataloader before state each epoch
    dataloader.initializeBatches()

    counter = 0
    while dataloader.hasNextBatch():
        batch = dataloader.getBatch()
        model.evaluate_batch(batch)

        counter += 1
        if counter % 100 == 0:
            print("Evaluated {} batches".format(counter))

    print("Evaluated {} batches".format(counter))

    model.reporter.report()


if __name__ == "__main__":
    dataset_dir = Path("fb15k_dataset/")
    dataset = FB15K237(dataset_dir)
    if not (dataset_dir / Path("edges/train_edges.bin")).exists():
        dataset.download()
        dataset.preprocess()

    dataset_stats = OmegaConf.load(dataset_dir / Path("dataset.yaml"))

    # create model
    device = torch.device("cpu")
    dtype = torch.float32
    embedding_dim = 50
    model, embeddings = init_model(embedding_dim, dataset_stats.num_nodes, dataset_stats.num_relations, device, dtype)

    # setup training dataloader
    train_edges = m.storage.tensor_from_file(
        filename=dataset.train_edges_file, shape=[dataset_stats.num_train, -1], dtype=torch.int32, device=device
    )
    train_neg_sampler = m.data.samplers.CorruptNodeNegativeSampler(
        num_chunks=10, num_negatives=500, degree_fraction=0.0, filtered=False
    )

    train_dataloader = m.data.DataLoader(
        edges=train_edges,
        node_embeddings=embeddings,
        batch_size=1000,
        neg_sampler=train_neg_sampler,
        learning_task="lp",
        train=True,
    )

    # setup eval dataloader
    valid_edges = m.storage.tensor_from_file(
        filename=dataset.valid_edges_file, shape=[dataset_stats.num_valid, -1], dtype=torch.int32, device=device
    )
    test_edges = m.storage.tensor_from_file(
        filename=dataset.test_edges_file, shape=[dataset_stats.num_test, -1], dtype=torch.int32, device=device
    )
    eval_neg_sampler = m.data.samplers.CorruptNodeNegativeSampler(filtered=True)

    eval_dataloader = m.data.DataLoader(
        edges=test_edges,
        node_embeddings=embeddings,
        batch_size=1000,
        neg_sampler=eval_neg_sampler,
        learning_task="lp",
        filter_edges=[train_edges, valid_edges],  # used to filter out false negatives in evaluation
        train=False,
    )

    for i in range(5):
        print("Train Epoch {}".format(i))
        print("-------------------------")
        train_epoch(model, train_dataloader)
        print("-------------------------")
        print("Evaluating")
        eval_epoch(model, eval_dataloader)
        print("-------------------------")
