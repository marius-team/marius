EdgeDecoder
=======================================

.. autoclass:: marius.nn.decoders.edge.EdgeDecoder
    :members:
    :undoc-members:
    :exclude-members: __init__, apply_relation, compute_scores, select_relations

    .. method:: __init__()
    
    .. method:: apply_relation(self: marius._nn.decoders.edge.EdgeDecoder, nodes: torch.Tensor, relations: torch.Tensor) -> torch.Tensor
    
    .. method:: compute_scores(self: marius._nn.decoders.edge.EdgeDecoder, src: torch.Tensor, dst: torch.Tensor) -> torch.Tensor
    
    .. method:: select_relations(self: marius._nn.decoders.edge.EdgeDecoder, indices: torch.Tensor, inverse: bool = False) -> torch.Tensor
