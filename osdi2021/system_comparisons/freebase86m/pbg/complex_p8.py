def get_torchbiggraph_config():

    config = dict(  # noqa
        # I/O data
        entity_path="freebase_8/freebase_metadata",
        edge_paths=[
            "freebase_8/freebase_train",
            "freebase_8/freebase_valid",
            "freebase_8/freebase_test",
        ],
        checkpoint_path="model/fb86m_8",
        # Graph structure
        entities={"all": {"num_partitions": 8}},
        relations=[
            {"name": "all_edges", "lhs": "all", "rhs": "all", "operator": "complex_diagonal"}
        ],
        dynamic_relations=True,
        # Scoring model
        dimension=50,
        global_emb=False,
        comparator="dot",
        # Training
        num_epochs=1,
        batch_size=10000,
        num_uniform_negs=500,
        num_batch_negs=500,
        loss_fn="softmax",
        lr=0.1,
        # Evaluation during training
        # GPU
        eval_fraction=0,
        verbose=0,
        bucket_order="affinity",
        num_gpus=1,
    )

    return config