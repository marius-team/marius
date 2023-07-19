from pathlib import Path

from omegaconf import OmegaConf

import marius as m
from marius.tools.preprocess.datasets.ogbn_arxiv import OGBNArxiv

import torch  # isort:skip


def init_model(feature_dim, num_classes, device):
    feature_layer = m.nn.layers.FeatureLayer(dimension=feature_dim, device=device)

    graph_sage_layer1 = m.nn.layers.GraphSageLayer(
        input_dim=feature_dim, output_dim=feature_dim, device=device, bias=True
    )

    graph_sage_layer2 = m.nn.layers.GraphSageLayer(
        input_dim=feature_dim, output_dim=feature_dim, device=device, bias=True
    )

    graph_sage_layer3 = m.nn.layers.GraphSageLayer(
        input_dim=feature_dim, output_dim=num_classes, device=device, bias=True
    )

    encoder = m.encoders.GeneralEncoder(
        layers=[[feature_layer], [graph_sage_layer1], [graph_sage_layer2], [graph_sage_layer3]]
    )

    # Setting up the decoder
    decoder = m.nn.decoders.node.NoOpNodeDecoder()

    # Loss Function
    loss = m.nn.CrossEntropyLoss(reduction="sum")

    # Set reporter to track accuracy at evaluation
    reporter = m.report.NodeClassificationReporter()
    reporter.add_metric(m.report.CategoricalAccuracy())

    # Making the model
    model = m.nn.Model(encoder, decoder, loss, reporter)

    # Set optimizer
    model.optimizers = [m.nn.AdamOptimizer(model.named_parameters(), lr=0.01)]

    return model


def train_epoch(model, dataloader):
    # need to reset dataloader state before each epoch
    dataloader.initializeBatches()

    counter = 0
    while dataloader.hasNextBatch():
        batch = dataloader.getBatch()
        model.train_batch(batch)

        counter += 1
        if counter % 50 == 0:
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
        if counter % 50 == 0:
            print("Evaluated {} batches".format(counter))

    print("Evaluated {} batches".format(counter))

    model.reporter.report()


if __name__ == "__main__":
    # initialize and preprocess dataset
    dataset_dir = Path("ogbn_arxiv_nc_dataset/")
    dataset = OGBNArxiv(dataset_dir)
    if not (dataset_dir / Path("edges/train_edges.bin")).exists():
        dataset.download()
        dataset.preprocess()

    dataset_stats = OmegaConf.load(dataset_dir / Path("dataset.yaml"))

    # Create the model
    device = torch.device("cuda")
    dtype = torch.float32
    feature_dim = dataset_stats.node_feature_dim
    model = init_model(feature_dim, dataset_stats.num_classes, device)

    # load training Data - Edges, Nodes, Features, labels
    edges_all = m.storage.tensor_from_file(
        filename=dataset.edge_list_file, shape=[dataset_stats.num_edges, -1], dtype=torch.int32, device=device
    )
    train_nodes = m.storage.tensor_from_file(
        filename=dataset.train_nodes_file, shape=[dataset_stats.num_train], dtype=torch.int32, device=device
    )
    features = m.storage.tensor_from_file(
        filename=dataset.node_features_file, shape=[dataset_stats.num_nodes, -1], dtype=torch.float32, device=device
    )
    labels = m.storage.tensor_from_file(
        filename=dataset.node_labels_file, shape=[dataset_stats.num_nodes], dtype=torch.int32, device=device
    )

    nbr_sampler_3_hop = m.data.samplers.LayeredNeighborSampler(num_neighbors=[-1, -1, -1])
    train_dataloader = m.data.DataLoader(
        nodes=train_nodes,
        edges=edges_all,
        node_features=features,
        node_labels=labels,
        batch_size=1000,
        nbr_sampler=nbr_sampler_3_hop,
        learning_task="nc",
        train=True,
    )

    # Evaluation:
    test_nodes = m.storage.tensor_from_file(
        filename=dataset.test_nodes_file, shape=[dataset_stats.num_test], dtype=torch.int32, device=device
    )
    eval_dataloader = m.data.DataLoader(
        nodes=test_nodes,
        edges=edges_all,
        node_labels=labels,
        node_features=features,
        batch_size=1000,
        nbr_sampler=nbr_sampler_3_hop,
        learning_task="nc",
        train=False,
    )

    # Doing the iterations
    for i in range(5):
        print("Train Epoch {}".format(i))
        print("-------------------------")
        train_epoch(model, train_dataloader)
        print()
        print("Evaluating")
        eval_epoch(model, eval_dataloader)

        print("-------------------------")
