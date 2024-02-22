GraphStorage
=======================================

.. autoclass:: marius.storage.GraphModelStorage
    :members:
    :undoc-members:
    :exclude-members: __init__, getNodeEmbeddingState, getNodeEmbeddingStateRange, getNodeEmbeddings, getNodeEmbeddingsRange, getNodeFeatures, getNodeFeaturesRange,  getNodeIdsRange, getNodeLabels, getNodeLabelsRange, getRandomNodeIds, get_edges, get_edges_range, init_subgraph, setActiveEdges, setActiveNodes, setBufferOrdering, updateAddNodeEmbeddingState, updateAddNodeEmbeddings, updatePutNodeEmbeddingState, updatePutNodeEmbeddings

    .. method:: __init__(self: marius._storage.GraphModelStorage, storage_ptrs: marius._storage.GraphModelStoragePtrs, storage_config: marius._config.StorageConfig) -> None
    
    .. method:: __init__(self: marius._storage.GraphModelStorage, edges: marius._storage.Storage, nodes: marius._storage.Storage = None, node_features: marius._storage.Storage = None, node_embeddings: marius._storage.Storage = None, node_optim_state: marius._storage.Storage = None, node_labels: marius._storage.Storage = None, filter_edges: List[marius._storage.Storage] = [], train: bool = False, prefetch: bool = False) -> None
    
    .. method:: getNodeEmbeddingState(self: marius._storage.GraphModelStorage, indices: torch.Tensor) -> torch.Tensor
    
    .. method:: getNodeEmbeddingStateRange(self: marius._storage.GraphModelStorage, start: int, size: int) -> torch.Tensor
    
    .. method:: getNodeEmbeddings(self: marius._storage.GraphModelStorage, indices: torch.Tensor) -> torch.Tensor
    
    .. method:: getNodeEmbeddingsRange(self: marius._storage.GraphModelStorage, start: int, size: int) -> torch.Tensor
    
    .. method:: getNodeFeatures(self: marius._storage.GraphModelStorage, indices: torch.Tensor) -> torch.Tensor
    
    .. method:: getNodeFeaturesRange(self: marius._storage.GraphModelStorage, start: int, size: int) -> torch.Tensor
    
    .. method:: getNodeIdsRange(self: marius._storage.GraphModelStorage, start: int, size: int) -> torch.Tensor
    
    .. method:: getNodeLabels(self: marius._storage.GraphModelStorage, indices: torch.Tensor) -> torch.Tensor
    
    .. method:: getNodeLabelsRange(self: marius._storage.GraphModelStorage, start: int, size: int) -> torch.Tensor

    .. method:: getRandomNodeIds(self: marius._storage.GraphModelStorage, size: int) -> torch.Tensor
    
    .. method:: get_edges(self: marius._storage.GraphModelStorage, indices: torch.Tensor) -> torch.Tensor
    
    .. method:: get_edges_range(self: marius._storage.GraphModelStorage, start: int, size: int) -> torch.Tensor
    
    .. method:: init_subgraph(self: marius._storage.GraphModelStorage, buffer_state: torch.Tensor) -> None
    
    .. method:: setActiveEdges(self: marius._storage.GraphModelStorage, active_edges: torch.Tensor) -> None
    
    .. method:: setActiveNodes(self: marius._storage.GraphModelStorage, node_ids: torch.Tensor) -> None
    
    .. method:: setBufferOrdering(self: marius._storage.GraphModelStorage, buffer_states: List[torch.Tensor]) -> None
    
    .. method:: updateAddNodeEmbeddingState(self: marius._storage.GraphModelStorage, indices: torch.Tensor, values: torch.Tensor) -> None
    
    .. method:: updateAddNodeEmbeddings(self: marius._storage.GraphModelStorage, indices: torch.Tensor, values: torch.Tensor) -> None
    
    .. method:: updatePutNodeEmbeddingState(self: marius._storage.GraphModelStorage, indices: torch.Tensor, state: torch.Tensor) -> None
    
    .. method:: updatePutNodeEmbeddings(self: marius._storage.GraphModelStorage, indices: torch.Tensor, embeddings: torch.Tensor) -> None