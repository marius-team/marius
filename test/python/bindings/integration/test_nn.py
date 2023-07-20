import unittest

from marius.config import LearningTask, LossOptions, LossReduction
from marius.data import Batch, DENSEGraph, MariusGraph
from marius.data.samplers import LayeredNeighborSampler
from marius.nn import CrossEntropyLoss, Model, SGDOptimizer, SoftmaxCrossEntropy
from marius.nn.decoders.edge import DistMult
from marius.nn.decoders.node import NoOpNodeDecoder
from marius.nn.encoders import GeneralEncoder
from marius.nn.layers import EmbeddingLayer
from marius.report import LinkPredictionReporter, NodeClassificationReporter

import torch  # isort:skip

edge_list = torch.tensor([[0, 0, 1], [0, 0, 2], [1, 1, 4], [2, 0, 3], [3, 1, 0], [4, 0, 1]])
batch_edges = torch.tensor(
    [
        [0, 0, 1],
        [2, 0, 3],
        [3, 1, 0],
    ]
)

node_ids = torch.tensor([0, 1, 2, 3])
node_embeddings = torch.tensor([[1.5, 2.5], [2.5, 3.5], [4.25, 1.0], [-1.0, 0.5]])

full_graph = MariusGraph(edge_list, edge_list[torch.argsort(edge_list[:, -1])], 5)
sampler = LayeredNeighborSampler(full_graph, [-1])
dense_graph = sampler.getNeighbors(node_ids)

num_relations = 2
embedding_dim = 2


def get_test_model_lp():
    device = torch.device("cpu")
    dtype = torch.float32

    embedding_layer = EmbeddingLayer(dimension=embedding_dim, device=device, bias=True)
    layers = [[embedding_layer]]
    encoder = GeneralEncoder(layers=layers)

    decoder = DistMult(
        num_relations=num_relations,
        embedding_dim=embedding_dim,
        use_inverse_relations=False,
        device=device,
        dtype=dtype,
        mode="infer",
    )

    loss = SoftmaxCrossEntropy(reduction="sum")

    reporter = LinkPredictionReporter()

    return Model(encoder, decoder, loss, reporter)


def get_test_model_lp_neg():
    device = torch.device("cpu")
    dtype = torch.float32

    embedding_layer = EmbeddingLayer(dimension=embedding_dim, device=device, bias=True)
    layers = [[embedding_layer]]
    encoder = GeneralEncoder(layers=layers)

    decoder = DistMult(
        num_relations=num_relations,
        embedding_dim=embedding_dim,
        use_inverse_relations=False,
        device=device,
        dtype=dtype,
        mode="train",
    )

    loss = SoftmaxCrossEntropy(reduction="sum")

    reporter = LinkPredictionReporter()

    return Model(encoder, decoder, loss, reporter)


def get_test_model_nc():
    device = torch.device("cpu")

    embedding_layer = EmbeddingLayer(dimension=embedding_dim, device=device, bias=True)
    layers = [[embedding_layer]]
    encoder = GeneralEncoder(layers=layers)

    decoder = NoOpNodeDecoder()

    loss = CrossEntropyLoss(reduction="sum")

    reporter = NodeClassificationReporter()

    return Model(encoder, decoder, loss, reporter)


class CustomModelBasic(Model):
    def __init__(self, encoder, decoder):
        if decoder.learning_task == LearningTask.LINK_PREDICTION:
            reporter = LinkPredictionReporter()
        else:
            reporter = NodeClassificationReporter()

        loss_options = LossOptions()
        loss_options.loss_reduction = LossReduction.SUM
        loss = SoftmaxCrossEntropy(reduction="sum")

        super().__init__(encoder, decoder, loss, reporter)


class CustomModelOverrideForward(Model):
    def __init__(self, encoder, decoder):
        if decoder.learning_task == LearningTask.LINK_PREDICTION:
            reporter = LinkPredictionReporter()
        else:
            reporter = NodeClassificationReporter()

        loss = SoftmaxCrossEntropy(reduction="sum")

        super().__init__(encoder, decoder, loss, reporter)

    def forward_lp(self, batch, train):
        pos = torch.ones([batch.edges.shape[0]])
        negs = torch.unsqueeze(torch.ones([batch.edges.shape[0]]), 0)
        return 3 * pos, 2 * negs, pos, 0 * negs


class TestModel(unittest.TestCase):
    """
    Model binding test
    """

    def test_construction_lp(self):
        get_test_model_lp()

    def test_construction_nc(self):
        get_test_model_nc()

    def test_forward_nc(self):
        model = get_test_model_nc()
        output = model.forward_nc(
            node_embeddings=node_embeddings, node_features=torch.empty([]), dense_graph=DENSEGraph(), train=False
        )
        assert torch.all(torch.eq(output, node_embeddings)).item() is True

    def test_forward_lp(self):
        model = get_test_model_lp()

        batch = Batch(False)

        batch.node_embeddings = node_embeddings
        batch.edges = batch_edges

        scores, _, _, _ = model.forward_lp(batch=batch, train=False)

        expected_scores = torch.tensor([12.5, -3.75, -0.25])

        assert torch.all(torch.eq(scores, expected_scores)).item() is True

    def test_train_batch(self):
        model_lp = get_test_model_lp_neg()

        batch = Batch(True)

        batch.node_embeddings = node_embeddings
        batch.node_embeddings_state = torch.zeros_like(node_embeddings)
        batch.edges = batch_edges
        batch.dst_neg_indices_mapping = torch.tensor([[2, 0], [0, 1], [1, 0]])

        model_lp.train_batch(batch, True)

        model_nc = get_test_model_nc()

        batch = Batch(True)

        batch.node_embeddings = node_embeddings
        batch.node_embeddings_state = torch.zeros_like(node_embeddings)
        batch.edges = batch_edges
        batch.node_labels = torch.tensor([0, 1, 0, 1], dtype=torch.long)

        model_nc.train_batch(batch, True)

    def test_clear_grad(self):
        model = get_test_model_nc()

        model.optimizers = [SGDOptimizer(model.named_parameters(), 0.1)]

        grad = torch.tensor([-1.0, -2.0])
        model.parameters()[0].grad = grad

        assert torch.all(torch.eq(model.parameters()[0].grad, grad)).item() is True
        model.clear_grad()
        assert model.parameters()[0].grad is None

    def test_step(self):
        model = get_test_model_nc()
        learning_rate = 0.1
        model.optimizers = [SGDOptimizer(model.named_parameters(), learning_rate)]

        grad = torch.tensor([-1.0, -2.0])
        model.parameters()[0].grad = grad

        assert torch.all(torch.eq(model.parameters()[0].grad, grad)).item() is True
        model.step()
        assert torch.all(torch.eq(model.parameters()[0], -grad * learning_rate)).item() is True

    def test_save(self):
        pass

    def test_load(self):
        pass

    def test_custom_model_basic(self):
        tmp_model = get_test_model_lp()
        model = CustomModelBasic(tmp_model.encoder, tmp_model.decoder)

        batch = Batch(False)

        batch.node_embeddings = node_embeddings
        batch.edges = batch_edges

        scores, _, _, _ = model.forward_lp(batch=batch, train=False)

        expected_scores = torch.tensor([12.5, -3.75, -0.25])

        assert torch.all(torch.eq(scores, expected_scores)).item() is True

    def test_custom_model_forward_override(self):
        tmp_model = get_test_model_lp_neg()
        model = CustomModelOverrideForward(tmp_model.encoder, tmp_model.decoder)

        batch = Batch(False)

        batch.node_embeddings = node_embeddings
        batch.edges = batch_edges

        scores1, scores2, scores3, scores4 = model.forward_lp(batch=batch, train=False)

        assert torch.all(torch.eq(scores1, 3 * torch.ones_like(scores1))).item() is True
        assert torch.all(torch.eq(scores2, 2 * torch.ones_like(scores1))).item() is True
        assert torch.all(torch.eq(scores3, 1 * torch.ones_like(scores1))).item() is True
        assert torch.all(torch.eq(scores4, 0 * torch.ones_like(scores1))).item() is True

    def test_init_from_config(self):
        pass
