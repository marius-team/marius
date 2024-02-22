FeatureLayer
=======================================

.. autoclass:: marius.nn.layers.FeatureLayer
    :members:
    :undoc-members:
    :exclude-members: __init__, forward

    .. method:: __init__(self: marius._nn.layers.FeatureLayer, layer_config: marius._config.LayerConfig, device: torch.device, offset: int = 0) -> None
    
    .. method:: __init__(self: marius._nn.layers.FeatureLayer, dimension: int, device: torch.device, bias: bool = False, bias_init: marius._config.InitConfig, activation: str = ‘none’, offset: int = 0) -> None

    .. method:: forward(self: marius._nn.layers.EmbeddingLayer, input: torch.Tensor) -> torch.Tensor