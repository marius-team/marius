GeneralEncoder
=======================================

.. autoclass:: marius.nn.encoders.GeneralEncoder
    :members:
    :undoc-members:
    :exclude-members: __init__, forward

    .. method:: __init__(self: marius._nn.encoders.GeneralEncoder, encoder_config: marius._config.EncoderConfig, device: torch.device, num_relations: int = 1) -> None
    
    .. method:: __init__(self: marius._nn.encoders.GeneralEncoder, layers: List[List[Layer]]) -> None
    
    .. method:: forward(self: marius._nn.encoders.GeneralEncoder, embeddings: Optional[torch.Tensor], features: Optional[torch.Tensor], dense_graph: marius._data.DENSEGraph, train: bool = True) -> torch.Tensor