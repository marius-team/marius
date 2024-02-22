GNNLayer
=======================================

.. autoclass:: marius.nn.layers.GNNLayer
    :members:
    :undoc-members:
    :exclude-members: __init__, forward

    .. method:: __init__()
    
    .. method:: forward(self: marius._nn.layers.GNNLayer, inputs: torch.Tensor, dense_graph: marius._data.DENSEGraph, train: bool) -> torch.Tensor

.. autoclass:: marius.nn.layers.GraphSageLayer
    :members:
    :undoc-members:
    :exclude-members: __init__, forward

    .. method:: __init__(self: marius._nn.layers.GraphSageLayer, layer_config: marius._config.LayerConfig, device: torch.device) -> None
    
    .. method:: __init__(self: marius._nn.layers.GraphSageLayer, input_dim: int, output_dim: int, device: Optional[torch.device] = None, aggregator: str = ‘mean’, init: marius._config.InitConfig, bias: bool = False, bias_init: marius._config.InitConfig, activation: str = ‘none’) -> None
    
    .. method:: forward(self: marius._nn.layers.GraphSageLayer, inputs: torch.Tensor, dense_graph: marius._data.DENSEGraph, train: bool = True) -> torch.Tensor

.. autoclass:: marius.nn.layers.GATLayer
    :members:
    :undoc-members:
    :exclude-members: __init__, forward

    .. method:: __init__(self: marius._nn.layers.GATLayer, layer_config: marius._config.LayerConfig, device: torch.device) -> None
    
    .. method:: __init__(self: marius._nn.layers.GATLayer, input_dim: int, output_dim: int, device: Optional[torch.device] = None, num_heads: int = 10, average_heads: bool = False, input_dropout: float = 0.0, attention_dropout: float = 0.0, negative_slope: float = 0.2, init: marius._config.InitConfig, bias: bool = False, bias_init: marius._config.InitConfig, activation: str = ‘none’) -> None
    
    .. method:: forward(self: marius._nn.layers.GATLayer, inputs: torch.Tensor, dense_graph: marius._data.DENSEGraph, train: bool = True) -> torch.Tensor

.. autoclass:: marius.nn.layers.GCNLayer
    :members:
    :undoc-members:
    :exclude-members: __init__, forward

    .. method:: __init__(self: marius._nn.layers.GCNLayer, layer_config: marius._config.LayerConfig, device: torch.device) -> None
    
    .. method:: __init__(self: marius._nn.layers.GCNLayer, input_dim: int, output_dim: int, device: Optional[torch.device] = None, init: marius._config.InitConfig, bias: bool = False, bias_init: marius._config.InitConfig, activation: str = ‘none’) -> None

    .. method:: forward(self: marius._nn.layers.GCNLayer, inputs: torch.Tensor, dense_graph: marius._data.DENSEGraph, train: bool = True) -> torch.Tensor