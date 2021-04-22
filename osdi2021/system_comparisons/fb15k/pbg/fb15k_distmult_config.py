def get_torchbiggraph_config():
    config = dict(  # noqa
        # I/O data
        entity_path="data/fb15k_distmult",
        edge_paths=[
            "data/FB15k/freebase_mtr100_mte100-train_partitioned",
            "data/FB15k/freebase_mtr100_mte100-valid_partitioned",
            "data/FB15k/freebase_mtr100_mte100-test_partitioned",
        ],
        checkpoint_path="model/fb15k_distmult",
        # Graph structure
        entities={"all": {"num_partitions": 1}},
        relations=[
            {"name": "all_edges", "lhs": "all", "rhs": "all", "operator": "diagonal"}
        ],
        dynamic_relations=True,
        # Scoring model
        dimension=400,
        global_emb=False,
        comparator="dot",
        # Training
        num_epochs=35,
        batch_size=10000,
        num_uniform_negs=500,
        num_batch_negs=500,
        loss_fn="softmax",
        regularization_coef=0,
        lr=0.1,
        # Evaluation during training
        # GPU
        eval_fraction=0,
        verbose=0,
        num_gpus=1,
    )

    return config