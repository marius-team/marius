#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE.txt file in the root directory of this source tree.

import argparse
from typing import Iterable, TextIO, BinaryIO
import torch
from torchbiggraph.checkpoint_manager import CheckpointManager
from torchbiggraph.config import ConfigFileLoader, ConfigSchema
from torchbiggraph.graph_storages import (
    ENTITY_STORAGES,
    RELATION_TYPE_STORAGES,
    EDGE_STORAGES,
    AbstractEntityStorage,
    AbstractRelationTypeStorage,
    AbstractEdgeStorage,
)
from torchbiggraph.model import MultiRelationEmbedder, make_model
import math



def preprocess_twitter_dglke():
    pass

def preprocess_twitter_pybg():
    pass

def preprocess_freebase86m_pybg():
    pass

def preprocess_livejournal_dglke():
    pass

def copy_and_evaluate_pybg():
    pass


def write(outf: TextIO, key: Iterable[str], value: Iterable[float]) -> None:
    outf.write("%s\t%s\n" % ("\t".join(key), "\t".join("%.9f" % x for x in value)))


def make_tsv(
        config: ConfigSchema,
        relations: bool
) -> None:
    entity_storage = ENTITY_STORAGES.make_instance(config.entity_path)
    test_edge_storage = EDGE_STORAGES.make_instance(config.edge_paths[0])

    model = make_model(config)

    checkpoint_manager = CheckpointManager(config.checkpoint_path)
    state_dict, _ = checkpoint_manager.read_model()
    if state_dict is not None:
        model.load_state_dict(state_dict, strict=False)

    for k, v in model.entities.items():
        num_partitions = v.num_partitions

    offsets = convert_embeddings(model, checkpoint_manager, entity_storage)
    convert_edges(test_edge_storage, num_partitions, offsets)

    if relations:
        relation_type_storage = RELATION_TYPE_STORAGES.make_instance(config.entity_path)
        convert_relations(model, relation_type_storage)


def convert_edges(
        edges: AbstractEdgeStorage,
        num_partitions: int,
        offsets: list
) -> None:

    with open("edges.bin", "wb") as f:
        for lhs_partition in range(num_partitions):
            for rhs_partition in range(num_partitions):
                lhs_offset = offsets[lhs_partition]
                rhs_offset = offsets[rhs_partition]
                edge_part = edges.load_edges(lhs_partition, rhs_partition)
                out_edges = torch.stack([edge_part.lhs.tensor + lhs_offset, edge_part.rel, edge_part.rhs.tensor + rhs_offset]).to(torch.int32).T
                tmp = out_edges.flatten(0, 1).numpy()
                f.write(bytes(tmp))


def convert_embeddings(
        model: MultiRelationEmbedder,
        checkpoint_manager: CheckpointManager,
        entity_storage: AbstractEntityStorage,
) -> list:

    offsets = []
    embedding_sizes = []
    with open("embeddings.bin", "wb") as f:
        for ent_t_name, ent_t_config in model.entities.items():
            for partition in range(ent_t_config.num_partitions):
                entities = entity_storage.load_names(ent_t_name, partition)
                embeddings, _ = checkpoint_manager.read(ent_t_name, partition)

                if model.global_embs is not None:
                    embeddings += model.global_embs[model.EMB_PREFIX + ent_t_name]

                embedding_sizes.append(embeddings.size(0))

                f.write(bytes(embeddings.flatten(0, 1).numpy()))

    for i, size in enumerate(embedding_sizes):
        if i == 0:
            offsets.append(0)
        else:
            prev = offsets[i - 1]
            offsets.append(prev + embedding_sizes[i - 1])

    return offsets


def convert_relations(
        model: MultiRelationEmbedder,
        relation_type_storage: AbstractRelationTypeStorage
) -> None:
    relation_types = relation_type_storage.load_names()
    with open("lhs_relations.bin", "wb") as f, open("rhs_relations.bin", "wb") as g:
        rel_t_config, = model.relations
        op_name = rel_t_config.operator
        lhs_operator, = model.lhs_operators
        rhs_operator, = model.rhs_operators
        lhs_real = None
        lhs_imag = None
        rhs_real = None
        rhs_imag = None
        for side, operator in [("lhs", lhs_operator), ("rhs", rhs_operator)]:
            for param_name, all_params in operator.named_parameters():
                if side == "lhs":
                    if param_name == "real":
                        lhs_real = all_params
                    else:
                        lhs_imag = all_params
                else:
                    if param_name == "real":
                        rhs_real = all_params
                    else:
                        rhs_imag = all_params

        lhs = torch.cat([lhs_real, lhs_imag], dim=1)
        rhs = torch.cat([rhs_real, rhs_imag], dim=1)
        f.write(bytes(lhs.detach().numpy()))
        g.write(bytes(rhs.detach().numpy()))