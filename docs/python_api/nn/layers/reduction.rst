ReductionLayer
=======================================

.. autoclass:: marius.nn.layers.ReductionLayer
    :members:
    :undoc-members:
    :exclude-members: __init__, forward

    .. method:: __init__()
    
    .. method:: forward(self: marius._nn.layers.ReductionLayer, inputs: List[torch.Tensor]) -> torch.Tensor

.. autoclass:: marius.nn.layers.ConcatReduction
    :members:
    :undoc-members:
    :exclude-members: __init__, forward

    .. method:: __init__(self: marius._nn.layers.ConcatReduction, layer_config: marius._config.LayerConfig, device: torch.device) -> None
    
    .. method:: __init__(self: marius._nn.layers.ConcatReduction, input_dim: int, output_dim: int, device: Optional[torch.device] = None, init: marius._config.InitConfig, bias: bool = False, bias_init: marius._config.InitConfig, activation: str = ‘none’) -> None
    
    .. method:: forward(self: marius._nn.layers.ConcatReduction, inputs: List[torch.Tensor]) -> torch.Tensor

.. autoclass:: marius.nn.layers.LinearReduction
    :members:
    :undoc-members:
    :exclude-members: __init__, forward

    .. method:: __init__(self: marius._nn.layers.LinearReduction, layer_config: marius._config.LayerConfig, device: torch.device) -> None
    
    .. method:: __init__(self: marius._nn.layers.LinearReduction, input_dim: int, output_dim: int, device: Optional[torch.device] = None, init: marius._config.InitConfig, bias: bool = False, bias_init: marius._config.InitConfig, activation: str = ‘none’) -> None
    
    .. method:: forward(self: marius._nn.layers.LinearReduction, inputs: List[torch.Tensor]) -> torch.Tensor