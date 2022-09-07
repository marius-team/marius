Initialization
=======================================

.. autofunction:: marius.nn.compute_fans

.. function:: marius.nn.glorot_uniform(shape: List[int], device: object, dtype: object, fans: Tuple[int, int] = (- 1, - 1)) -> torch.Tensor

.. function:: marius.nn.glorot_normal(shape: List[int], device: object, dtype: object, fans: Tuple[int, int] = (- 1, - 1)) -> torch.Tensor

.. function:: marius.nn.constant_init(shape: List[int], constant: float = 0, device: object, dtype: object) -> torch.Tensor

.. function:: marius.nn.uniform_init(shape: List[int], scale_factor: float = 0.001, device: object, dtype: object) -> torch.Tensor

.. function:: marius.nn.normal_init(shape: List[int], mean: float = 0, std: float = 1, device: object, dtype: object) -> torch.Tensor

.. function:: marius.nn.initialize_tensor(init_config: marius._config.InitConfig, shape: List[int], device: object, dtype: object, fans: Tuple[int, int] = (- 1, - 1)) -> torch.Tensor

.. function:: marius.nn.initialize_subtensor(init_config: marius._config.InitConfig, sub_shape: List[int], full_shape: List[int], device: object, dtype: object, fans: Tuple[int, int] = (- 1, - 1)) -> torch.Tensor
