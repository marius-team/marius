
MariusGraph
============================


.. autoclass:: marius.data.MariusGraph
    :members:
    :undoc-members:
    :exclude-members: __init__, getEdges, getNeighborOffsets, getNeighborsForNodeIds, getNumNeighbors, getRelationIDs, to

    .. method:: __init__(self: marius._data.MariusGraph) -> None
    
    .. method:: __init__(self: marius._data.MariusGraph, src_sorted_edges: torch.Tensor, dst_sorted_edges: torch.Tensor, num_nodes_in_memory: int) -> None
    
    .. method:: getEdges(self: marius._data.MariusGraph, incoming: bool = True) -> torch.Tensor
    
    .. method:: getNeighborOffsets(self: marius._data.MariusGraph, incoming: bool = True) -> torch.Tensor
    
    .. method:: getNeighborsForNodeIds(self: marius._data.MariusGraph, node_ids: torch.Tensor, incoming: bool, neighbor_sampling_layer: marius._config.NeighborSamplingLayer, max_neighbors_size: int, rate: float) -> Tuple(torch.Tensor, torch.Tensor)
    
    .. method:: getNumNeighbors(self: marius._data.MariusGraph, incoming: bool = True) -> torch.Tensor
    
    .. method:: getRelationIDs(self: marius._data.MariusGraph, incoming: bool = True) -> torch.Tensor
    
    .. method:: to(self: marius._data.MariusGraph, device: torch.device) -> None