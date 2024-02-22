Optimizers
********************

.. autoclass:: marius.nn.Optimizer
    :members:
    :undoc-members:
    :exclude-members: __init__

    .. method:: __init__()

.. autoclass:: marius.nn.SGDOptimizer
    :members:
    :undoc-members:
    :special-members: __init__

.. autoclass:: marius.nn.AdagradOptimizer
    :members:
    :undoc-members:
    :exclude-members: __init__

    .. method:: __init__(self: marius._nn.AdagradOptimizer, param_dict: torch._C.cpp.OrderedTensorDict, options: marius._config.AdagradOptions) -> None
    
    .. method:: __init__(self: marius._nn.AdagradOptimizer, param_dict: torch._C.cpp.OrderedTensorDict, lr: float = 0.1, eps: float = 1e-10, lr_decay: float = 0, init_value: float = 0, weight_decay: float = 0) -> None

.. autoclass:: marius.nn.AdamOptimizer
    :members:
    :undoc-members:
    :exclude-members: __init__

    .. method:: __init__(self: marius._nn.AdamOptimizer, param_dict: torch._C.cpp.OrderedTensorDict, options: marius._config.AdamOptions) -> None
    
    .. method:: __init__(self: marius._nn.AdamOptimizer, param_dict: torch._C.cpp.OrderedTensorDict, lr: float = 0.1, eps: float = 1e-08, beta_1: float = 0.9, beta_2: float = 0.999, weight_decay: float = 0, amsgrad: bool = False) -> None