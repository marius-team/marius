
DENSEGraph
============================


.. autoclass:: marius.data.DENSEGraph
    :members:
    :undoc-members:
    :exclude-members: __init__, getNeighborIDs, setNodeProperties, to

    .. method:: __init__(self: marius._data.DENSEGraph) -> None
    
    .. method:: __init__(self: marius._data.DENSEGraph, hop_offsets: torch.Tensor, node_ids: torch.Tensor, in_offsets: torch.Tensor, in_neighbors_vec: List[torch.Tensor], in_neighbors: torch.Tensor, out_offsets: torch.Tensor, out_neighbors_vec: List[torch.Tensor], out_neighbors: torch.Tensor, num_nodes_in_memory: int) -> None
    
    .. method:: getNeighborIDs(self: marius._data.DENSEGraph, incoming: bool = True, global_ids: bool = False) -> torch.Tensor
    
    .. method:: setNodeProperties(self: marius._data.DENSEGraph, node_properties: torch.Tensor) -> None
    
    .. method:: to(self: marius._data.DENSEGraph, device: torch.device) -> None