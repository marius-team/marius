def get_torchbiggraph_config():

    config = dict(  # noqa
        # I/O data
        entity_path="osdi2021/pybg_converter/pbg-helper/dataset_importer/output_data/twitter16/twitter_metadata",
        edge_paths=[
            "osdi2021/pybg_converter/pbg-helper/dataset_importer/output_data/twitter16/twitter_train",
            "osdi2021/pybg_converter/pbg-helper/dataset_importer/output_data/twitter16/twitter_valid",
            "osdi2021/pybg_converter/pbg-helper/dataset_importer/output_data/twitter16/twitter_test",
        ],
        checkpoint_path="model/twitter16",
        # Graph structure
        entities={"all": {"num_partitions": 16}},
        relations=[
            {"name": "all_edges", "lhs": "all", "rhs": "all", "operator": "none"}
        ],
        dynamic_relations=False,
        # Scoring model
        dimension=100,
        global_emb=False,
        comparator="dot",
        bucket_order="affinity",
        # Training
        num_epochs=10,
        batch_size=50000,
        num_uniform_negs=500,
        num_batch_negs=500,
        loss_fn="softmax",
        lr=0.1,
        # Evaluation during training
        eval_fraction=0,
        # GPU
        verbose=1,
        num_gpus=1,
    )

    return config