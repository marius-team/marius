EmbeddingLayer
=======================================

.. autoclass:: marius.nn.layers.EmbeddingLayer
    :members:
    :undoc-members:
    :exclude-members: __init__, forward, init_embeddings

    .. method:: __init__(self: marius._nn.layers.EmbeddingLayer, layer_config: marius._config.LayerConfig, device: torch.device, offset: int = 0) -> None
    
    .. method:: __init__(self: marius._nn.layers.EmbeddingLayer, dimension: int, device: torch.device, init: marius._config.InitConfig, bias: bool = False, bias_init: marius._config.InitConfig, activation: str = ‘none’, offset: int = 0) -> None
    
    .. method:: forward(self: marius._nn.layers.EmbeddingLayer, input: torch.Tensor) -> torch.Tensor
    
    .. method:: init_embeddings(self: marius._nn.layers.EmbeddingLayer, num_nodes: int) -> torch.Tensor