def get_torchbiggraph_config():

    config = dict(  # noqa
        # I/O data
        entity_path="data/lj",
        edge_paths=[
            "data/lj/train",
            "data/lj/valid",
            "data/lj/test",
        ],
        checkpoint_path="model/lj",
        # Graph structure
        entities={"all": {"num_partitions": 1}},
        relations=[
            {"name": "all_edges", "lhs": "all", "rhs": "all", "operator": "none"}
        ],
        dynamic_relations=False,
        # Scoring model
        dimension=100,
        global_emb=False,
        comparator="dot",
        # Training
        num_epochs=25,
        batch_size=50000,
        num_uniform_negs=500,
        num_batch_negs=500,
        loss_fn="softmax",
        lr=0.1,
        # Evaluation during training
        eval_fraction=0,
        # GPU
        num_gpus=1,
    )

    return config