Model
********************

.. autoclass:: marius.nn.Model
    :members:
    :undoc-members:
    :exclude-members: __init__, broadcast, forward_lp, forward_nc

    .. method:: __init__(self: marius._nn.Model, arg0: GeneralEncoder, arg1: Decoder, arg2: marius._nn.LossFunction, arg3: Reporter) -> None
    
    .. method:: __init__(self: marius._nn.Model, encoder: GeneralEncoder, decoder: Decoder, loss: marius._nn.LossFunction = None, reporter: Reporter = None, sparse_lr: float = 0.1) -> None
    
    .. method:: broadcast(self: marius._nn.Model, devices: List[torch.device]) -> None
    
    .. method:: forward_lp(self: marius._nn.Model, batch: marius._data.Batch, train: bool) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
    
    .. method:: forward_nc(self: marius._nn.Model, node_embeddings: Optional[torch.Tensor], node_features: Optional[torch.Tensor], dense_graph: marius._data.DENSEGraph, train: bool) -> torch.Tensor