
DataLoader
============================


.. autoclass:: marius.data.DataLoader
    :members:
    :undoc-members:
    :exclude-members: __init__, getBatch

    .. method:: __init__(self: marius._data.DataLoader, graph_storage: GraphModelStorage, learning_task: str, training_config: marius._config.TrainingConfig, evaluation_config: marius._config.EvaluationConfig, encoder_config: marius._config.EncoderConfig) -> None
    
    .. method:: __init__(self: marius._data.DataLoader, graph_storage: GraphModelStorage, learning_task: str, batch_size: int = 1000, neg_sampler: marius._data.samplers.NegativeSampler = None, nbr_sampler: marius._data.samplers.NeighborSampler = None, train: bool = False) -> None
    
    .. method:: __init__(self: marius._data.DataLoader, edges: Optional[torch.Tensor], learning_task: str, nodes: Optional[torch.Tensor] = None, node_features: Optional[torch.Tensor] = None, node_embeddings: Optional[torch.Tensor] = None, node_optim_state: Optional[torch.Tensor] = None, node_labels: Optional[torch.Tensor] = None, train_edges: Optional[torch.Tensor] = None, batch_size: int = 1000, neg_sampler: marius._data.samplers.NegativeSampler = None, nbr_sampler: marius._data.samplers.NeighborSampler = None, filter_edges: List[torch.Tensor] = [], train: bool = False) -> None
    
    .. method:: getBatch(self: marius._data.DataLoader, device: Optional[torch.device] = None, perform_map: bool = True) -> marius._data.Batch