def get_torchbiggraph_config():

    config = dict(  # noqa
        # I/O data
        entity_path="/mnt/mount_point/freebase_8/freebase_metadata",
        edge_paths=[
            "/mnt/mount_point/freebase_8/freebase_train",
            "/mnt/mount_point/freebase_8/freebase_valid",
            "/mnt/mount_point/freebase_8/freebase_test",
        ],
        checkpoint_path="/mnt/mount_point/model/fb86m_8",
        # Graph structure
        entities={"all": {"num_partitions": 8}},
        relations=[
            {"name": "all_edges", "lhs": "all", "rhs": "all", "operator": "complex_diagonal"}
        ],
        dynamic_relations=True,
        # Scoring model
        dimension=100,
        global_emb=False,
        comparator="dot",
        # Training
        num_epochs=10,
        batch_size=10000,
        num_uniform_negs=500,
        num_batch_negs=500,
        loss_fn="softmax",
        lr=0.1,
        num_machines=4,
        distributed_init_method='tcp://172.31.20.89:30050',
        # Evaluation during training
        eval_fraction=0,
        verbose=0,
    )

    return config
