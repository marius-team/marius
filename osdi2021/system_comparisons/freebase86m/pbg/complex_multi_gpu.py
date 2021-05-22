def get_torchbiggraph_config():

    config = dict(  # noqa
        # I/O data
        entity_path="freebase_16/freebase_metadata",
        edge_paths=[
            "freebase_16/freebase_train",
            "freebase_16/freebase_valid",
            "freebase_16/freebase_test",
        ],
        checkpoint_path="model/fb86m_16",
        # Graph structure
        entities={"all": {"num_partitions": 16}},
        relations=[
            {"name": "all_edges", "lhs": "all", "rhs": "all", "operator": "complex_diagonal"}
        ],
        dynamic_relations=True,
        # Scoring model
        dimension=50,
        global_emb=False,
        comparator="dot",
        # Training
        num_epochs=10,
        batch_size=50000,
        num_uniform_negs=500,
        num_batch_negs=500,
        loss_fn="softmax",
        lr=0.1,
        # Evaluation during training
        # GPU
        eval_fraction=0,
        verbose=0,
        num_gpus=8,
    )

    return config