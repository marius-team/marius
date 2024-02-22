from pathlib import Path

import numpy as np
import pandas as pd
from omegaconf import OmegaConf

import marius as m
from marius.tools.configuration.constants import PathConstants
from marius.tools.preprocess.converters.torch_converter import TorchEdgeListConverter
from marius.tools.preprocess.dataset import NodeClassificationDataset
from marius.tools.preprocess.datasets.dataset_helpers import remap_nodes
from marius.tools.preprocess.utils import download_url, extract_file

import torch  # isort:skip


def switch_to_num(row):
    names = [
        "Neural_Networks",
        "Rule_Learning",
        "Reinforcement_Learning",
        "Probabilistic_Methods",
        "Theory",
        "Genetic_Algorithms",
        "Case_Based",
    ]
    idx = 0
    for i in range(len(names)):
        if row == names[i]:
            idx = i
            break

    return idx


class MYDATASET(NodeClassificationDataset):
    def __init__(self, output_directory: Path, spark=False):
        super().__init__(output_directory, spark)

        self.dataset_name = "cora"
        self.dataset_url = "http://www.cs.umd.edu/~sen/lbc-proj/data/cora.tgz"

    def download(self, overwrite=False):
        # These are the files we want to make my the end of the the download
        self.input_edge_list_file = self.output_directory / Path("edge.csv")
        self.input_node_feature_file = self.output_directory / Path("node-feat.csv")
        self.input_node_label_file = self.output_directory / Path("node-label.csv")
        self.input_train_nodes_file = self.output_directory / Path("train.csv")
        self.input_valid_nodes_file = self.output_directory / Path("valid.csv")
        self.input_test_nodes_file = self.output_directory / Path("test.csv")

        # If files already exist we don't need to do processing
        download = False
        if not self.input_edge_list_file.exists():
            download = True
        if not self.input_node_feature_file.exists():
            download = True
        if not self.input_node_label_file.exists():
            download = True
        if not self.input_train_nodes_file.exists():
            download = True
        if not self.input_valid_nodes_file.exists():
            download = True
        if not self.input_test_nodes_file.exists():
            download = True

        if download:
            archive_path = download_url(self.dataset_url, self.output_directory, overwrite)
            extract_file(archive_path, remove_input=False)

            # Reading and processing the csv
            df = pd.read_csv(dataset_dir / Path("cora/cora.content"), sep="\t", header=None)
            cols = df.columns[1 : len(df.columns) - 1]

            # Getting all the indices
            indices = np.array(range(len(df)))
            np.random.shuffle(indices)
            train_indices = indices[0 : int(0.8 * len(df))]
            valid_indices = indices[int(0.8 * len(df)) : int(0.8 * len(df)) + int(0.1 * len(df))]
            test_indices = indices[int(0.8 * len(df)) + int(0.1 * len(df)) :]

            np.savetxt(dataset_dir / Path("train.csv"), train_indices, delimiter=",", fmt="%d")
            np.savetxt(dataset_dir / Path("valid.csv"), valid_indices, delimiter=",", fmt="%d")
            np.savetxt(dataset_dir / Path("test.csv"), test_indices, delimiter=",", fmt="%d")

            # Features
            features = df[cols]
            features.to_csv(index=False, sep=",", path_or_buf=dataset_dir / Path("node-feat.csv"), header=False)

            # Labels
            labels = df[df.columns[len(df.columns) - 1]]
            labels = labels.apply(switch_to_num)
            labels.to_csv(index=False, sep=",", path_or_buf=dataset_dir / Path("node-label.csv"), header=False)

            # Edges
            node_ids = df[df.columns[0]]
            dict_reverse = node_ids.to_dict()
            nodes_dict = {v: k for k, v in dict_reverse.items()}
            df_edges = pd.read_csv(dataset_dir / Path("cora/cora.cites"), sep="\t", header=None)
            df_edges.replace({0: nodes_dict, 1: nodes_dict}, inplace=True)
            df_edges.to_csv(index=False, sep=",", path_or_buf=dataset_dir / Path("edge.csv"), header=False)

    def preprocess(
        self, num_partitions=1, remap_ids=True, splits=None, sequential_train_nodes=False, partitioned_eval=False
    ):
        train_nodes = np.genfromtxt(self.input_train_nodes_file, delimiter=",").astype(np.int32)
        valid_nodes = np.genfromtxt(self.input_valid_nodes_file, delimiter=",").astype(np.int32)
        test_nodes = np.genfromtxt(self.input_test_nodes_file, delimiter=",").astype(np.int32)

        # Calling the convert function to generate the preprocessed files
        converter = TorchEdgeListConverter(
            output_dir=self.output_directory,
            train_edges=self.input_edge_list_file,
            num_partitions=num_partitions,
            src_column=0,
            dst_column=1,
            remap_ids=remap_ids,
            sequential_train_nodes=sequential_train_nodes,
            delim=",",
            known_node_ids=[train_nodes, valid_nodes, test_nodes],
            partitioned_evaluation=partitioned_eval,
        )
        dataset_stats = converter.convert()

        features = np.genfromtxt(self.input_node_feature_file, delimiter=",").astype(np.float32)
        labels = np.genfromtxt(self.input_node_label_file, delimiter=",").astype(np.int32)

        # The remap in the convertor will only change the edge.csv so we need to manually
        # remap rest of the *.csv files. We are doing that here
        if remap_ids:
            node_mapping = np.genfromtxt(self.output_directory / Path(PathConstants.node_mapping_path), delimiter=",")
            train_nodes, valid_nodes, test_nodes, features, labels = remap_nodes(
                node_mapping, train_nodes, valid_nodes, test_nodes, features, labels
            )

        # Writing the remapped files as bin files
        with open(self.train_nodes_file, "wb") as f:
            f.write(bytes(train_nodes))
        with open(self.valid_nodes_file, "wb") as f:
            f.write(bytes(valid_nodes))
        with open(self.test_nodes_file, "wb") as f:
            f.write(bytes(test_nodes))
        with open(self.node_features_file, "wb") as f:
            f.write(bytes(features))
        with open(self.node_labels_file, "wb") as f:
            f.write(bytes(labels))

        # update dataset yaml
        dataset_stats.num_train = train_nodes.shape[0]
        dataset_stats.num_valid = valid_nodes.shape[0]
        dataset_stats.num_test = test_nodes.shape[0]
        dataset_stats.node_feature_dim = features.shape[1]
        dataset_stats.num_classes = 40

        dataset_stats.num_nodes = dataset_stats.num_train + dataset_stats.num_valid + dataset_stats.num_test

        with open(self.output_directory / Path("dataset.yaml"), "w") as f:
            yaml_file = OmegaConf.to_yaml(dataset_stats)
            f.writelines(yaml_file)

        return


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
    # Here we are initializing the cora dataset. Details regarding what this
    # dataset class is doing can be found: [TODO add path location]

    # initialize and preprocess dataset
    dataset_dir = Path("cora/")
    dataset = MYDATASET(dataset_dir)
    if not (dataset_dir / Path("edges/train_edges.bin")).exists():
        dataset.download()
        dataset.preprocess()

    dataset_stats = OmegaConf.load(dataset_dir / Path("dataset.yaml"))

    # Rest of the code (i.e. model, dataloader, training, etc) is same as nc_ogbn_arxiv example.
    # Please refer to the documentation at docs/examples/python/nc_ogbn_arxiv.rst for details rest of the code

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
