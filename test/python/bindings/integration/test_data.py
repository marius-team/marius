import unittest

from marius.data import Batch, DataLoader
from marius.data.samplers import CorruptNodeNegativeSampler, LayeredNeighborSampler

import torch  # isort:skip


class TestBatch(unittest.TestCase):
    """
    Batch binding tests
    """

    def test_construction(self):
        b1 = Batch(train=False)

        assert b1.node_embeddings is None
        assert b1.train is False
        assert b1.device_id == -1

        rand_tens = torch.randn([10])
        b1.node_embeddings = rand_tens

        assert torch.all(torch.eq(b1.node_embeddings, rand_tens)).item() is True
        b2 = Batch(train=True)

        assert b2.node_embeddings is None
        assert b2.train is True
        assert b2.device_id == -1

        b2.node_embeddings = rand_tens
        assert torch.all(torch.eq(b2.node_embeddings, rand_tens)).item() is True

    def test_accumulate_gradients(self):
        b = Batch(train=True)

        b.node_embeddings = torch.tensor([2.0, 4.0])
        b.node_embeddings.grad = torch.tensor([0.5, -1.0])
        b.node_embeddings_state = torch.tensor([0.0, 0.0])

        b.accumulateGradients(learning_rate=1.0)

        assert b.node_embeddings_state is None
        assert torch.all(torch.eq(b.node_state_update, b.node_embeddings.grad.pow(2))).item() is True

        expected = -1.0 * (b.node_embeddings.grad / (b.node_state_update.sqrt().add_(1e-10)))
        assert torch.all(torch.eq(b.node_gradients, expected)).item() is True

    def test_clear(self):
        b = Batch(train=True)

        b.node_embeddings = torch.tensor([2.0, 4.0])
        b.node_embeddings.grad = torch.tensor([0.5, -1.0])
        b.node_embeddings_state = torch.tensor([0.0, 0.0])

        assert b.node_embeddings is not None
        assert b.node_embeddings is not None
        assert b.node_embeddings_state is not None

        b.clear()

        assert b.node_embeddings is None
        assert b.node_embeddings is None
        assert b.node_embeddings_state is None


class TestDataloader(unittest.TestCase):
    def test_lp_only_edges(self):
        num_edges = 100
        num_nodes = 10
        d = 5

        batch_size = 10

        edges = torch.randint(0, num_nodes, size=(num_edges, 2))
        embeddings = torch.randn(size=(num_nodes, d))
        features = torch.randn(size=(num_nodes, d))

        neg_sampler = None
        nbr_sampler = None

        # constructor for in memory objects and tensors
        dataloader = DataLoader(
            edges=edges,
            node_embeddings=embeddings,
            node_features=features,
            batch_size=batch_size,
            neg_sampler=neg_sampler,
            nbr_sampler=nbr_sampler,
            learning_task="lp",
            train=False,
        )

        dataloader.initializeBatches()

        count = 0
        while dataloader.hasNextBatch():
            b = dataloader.getBatch()

            assert b.edges.shape[0] == batch_size
            assert b.unique_node_indices.shape[0] == b.node_embeddings.shape[0]
            assert b.unique_node_indices.shape[0] == b.node_features.shape[0]

            count += 1

        assert count == (num_edges / batch_size)

    def test_lp_negs(self):
        num_edges = 100
        num_nodes = 10
        d = 5

        batch_size = 10

        edges = torch.randint(0, num_nodes, size=(num_edges, 2))
        embeddings = torch.randn(size=(num_nodes, d))
        features = torch.randn(size=(num_nodes, d))

        num_chunks = 2
        num_negatives = 4
        degree_fraction = 0.5

        neg_sampler = CorruptNodeNegativeSampler(
            num_chunks=num_chunks, num_negatives=num_negatives, degree_fraction=degree_fraction, filtered=False
        )
        nbr_sampler = None

        # constructor for in memory objects and tensors
        dataloader = DataLoader(
            edges=edges,
            node_embeddings=embeddings,
            node_features=features,
            batch_size=batch_size,
            neg_sampler=neg_sampler,
            nbr_sampler=nbr_sampler,
            learning_task="lp",
            train=False,
        )

        dataloader.initializeBatches()

        count = 0
        while dataloader.hasNextBatch():
            b = dataloader.getBatch()

            assert b.edges.shape[0] == batch_size
            assert b.unique_node_indices.shape[0] == b.node_embeddings.shape[0]
            assert b.unique_node_indices.shape[0] == b.node_features.shape[0]

            assert b.src_neg_indices.shape[0] == num_chunks
            assert b.src_neg_indices.shape[1] == num_negatives
            assert b.dst_neg_indices.shape[0] == num_chunks
            assert b.dst_neg_indices.shape[1] == num_negatives

            assert b.src_neg_indices_mapping.shape[0] == num_chunks
            assert b.src_neg_indices_mapping.shape[1] == num_negatives
            assert b.dst_neg_indices_mapping.shape[0] == num_chunks
            assert b.dst_neg_indices_mapping.shape[1] == num_negatives

            count += 1

        assert count == (num_edges / batch_size)

    def test_lp_negs_nbrs(self):
        num_edges = 100
        num_nodes = 10
        d = 5

        batch_size = 10

        edges = torch.randint(0, num_nodes, size=(num_edges, 2))
        embeddings = torch.randn(size=(num_nodes, d))
        features = torch.randn(size=(num_nodes, d))

        num_chunks = 2
        num_negatives = 4
        degree_fraction = 0.5

        neg_sampler = CorruptNodeNegativeSampler(
            num_chunks=num_chunks, num_negatives=num_negatives, degree_fraction=degree_fraction, filtered=False
        )
        nbr_sampler = LayeredNeighborSampler([-1])

        # constructor for in memory objects and tensors
        dataloader = DataLoader(
            edges=edges,
            node_embeddings=embeddings,
            node_features=features,
            batch_size=batch_size,
            neg_sampler=neg_sampler,
            nbr_sampler=nbr_sampler,
            learning_task="lp",
            train=False,
        )

        dataloader.initializeBatches()

        count = 0
        while dataloader.hasNextBatch():
            b = dataloader.getBatch()

            assert b.edges.shape[0] == batch_size
            assert b.unique_node_indices.shape[0] == b.node_embeddings.shape[0]
            assert b.unique_node_indices.shape[0] == b.node_features.shape[0]

            assert b.src_neg_indices.shape[0] == num_chunks
            assert b.src_neg_indices.shape[1] == num_negatives
            assert b.dst_neg_indices.shape[0] == num_chunks
            assert b.dst_neg_indices.shape[1] == num_negatives

            assert b.src_neg_indices_mapping.shape[0] == num_chunks
            assert b.src_neg_indices_mapping.shape[1] == num_negatives
            assert b.dst_neg_indices_mapping.shape[0] == num_chunks
            assert b.dst_neg_indices_mapping.shape[1] == num_negatives

            assert torch.all(torch.eq(b.unique_node_indices, b.dense_graph.node_ids)).item() is True

            count += 1

        assert count == (num_edges / batch_size)

    def test_lp_nbrs(self):
        num_edges = 100
        num_nodes = 10
        d = 5

        batch_size = 10

        edges = torch.randint(0, num_nodes, size=(num_edges, 2))
        embeddings = torch.randn(size=(num_nodes, d))
        features = torch.randn(size=(num_nodes, d))

        neg_sampler = None
        nbr_sampler = LayeredNeighborSampler([-1])

        # constructor for in memory objects and tensors
        dataloader = DataLoader(
            edges=edges,
            node_embeddings=embeddings,
            node_features=features,
            batch_size=batch_size,
            neg_sampler=neg_sampler,
            nbr_sampler=nbr_sampler,
            learning_task="lp",
            train=False,
        )

        dataloader.initializeBatches()

        count = 0
        while dataloader.hasNextBatch():
            b = dataloader.getBatch()

            assert b.edges.shape[0] == batch_size
            assert b.unique_node_indices.shape[0] == b.node_embeddings.shape[0]
            assert b.unique_node_indices.shape[0] == b.node_features.shape[0]

            assert torch.all(torch.eq(b.unique_node_indices, b.dense_graph.node_ids)).item() is True

            count += 1

        assert count == (num_edges / batch_size)

    def test_nc_nbrs(self):
        num_edges = 100
        num_nodes = 50
        d = 5

        num_train = 25
        batch_size = 5

        edges = torch.randint(0, num_nodes, size=(num_edges, 2))
        embeddings = torch.randn(size=(num_nodes, d))
        features = torch.randn(size=(num_nodes, d))

        nodes = torch.arange(0, num_train)

        nbr_sampler = LayeredNeighborSampler([-1])

        # constructor for in memory objects and tensors
        dataloader = DataLoader(
            edges=edges,
            nodes=nodes,
            node_embeddings=embeddings,
            node_features=features,
            batch_size=batch_size,
            nbr_sampler=nbr_sampler,
            learning_task="nc",
            train=False,
        )

        dataloader.initializeBatches()

        count = 0
        while dataloader.hasNextBatch():
            b = dataloader.getBatch()

            assert b.unique_node_indices.shape[0] == b.node_embeddings.shape[0]
            assert b.unique_node_indices.shape[0] == b.node_features.shape[0]

            assert torch.all(torch.eq(b.unique_node_indices, b.dense_graph.node_ids)).item() is True

            count += 1

        assert count == (num_train / batch_size)

    def test_nc_no_nbrs(self):
        num_edges = 100
        num_nodes = 50
        d = 5

        num_train = 25
        batch_size = 5

        edges = torch.randint(0, num_nodes, size=(num_edges, 2))
        embeddings = torch.randn(size=(num_nodes, d))
        features = torch.randn(size=(num_nodes, d))

        nodes = torch.arange(0, num_train)

        # constructor for in memory objects and tensors
        dataloader = DataLoader(
            edges=edges,
            nodes=nodes,
            node_embeddings=embeddings,
            node_features=features,
            batch_size=batch_size,
            learning_task="nc",
            train=False,
        )

        dataloader.initializeBatches()

        count = 0
        while dataloader.hasNextBatch():
            b = dataloader.getBatch()

            assert b.unique_node_indices.shape[0] == b.node_embeddings.shape[0]
            assert b.unique_node_indices.shape[0] == b.node_features.shape[0]

            assert torch.all(torch.eq(b.unique_node_indices, b.root_node_indices)).item() is True

            count += 1

        assert count == (num_train / batch_size)
