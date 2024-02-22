Storage
=======================================

.. function:: marius.storage.tensor_from_file(filename: str, shape: List[int], dtype: torch.dtype, device: torch.device) -> torch.Tensor

.. autoclass:: marius.storage.Storage
    :members:
    :undoc-members:
    :exclude-members: __init__, indexAdd, indexPut, indexRead, range, rangePut

    .. method:: __init__()

    .. method:: indexAdd(self: marius._storage.Storage, indices: torch.Tensor, values: torch.Tensor) -> None

    .. method:: indexPut(self: marius._storage.Storage, indices: torch.Tensor, values: torch.Tensor) -> None

    .. method:: indexRead(self: marius._storage.Storage, indices: torch.Tensor) -> torch.Tensor

    .. method:: range(self: marius._storage.Storage, offset: int, n: int) -> torch.Tensor

    .. method:: rangePut(self: marius._storage.Storage, offset: int, n: int, values: torch.Tensor) -> None

.. autoclass:: marius.storage.FlatFile
    :members:
    :undoc-members:
    :exclude-members: __init__, append

    .. method:: __init__(self: marius._storage.FlatFile, filename: str, shape: List[int], dtype: torch.dtype, alloc: bool = False) -> None
    
    .. method:: __init__(self: marius._storage.FlatFile, filename: str, data: torch.Tensor) -> None
    
    .. method:: __init__(self: marius._storage.FlatFile, filename: str, dtype: torch.dtype) -> None
    
    .. method:: append(self: marius._storage.FlatFile, values: torch.Tensor) -> None

.. autoclass:: marius.storage.PartitionBufferStorage
    :members:
    :undoc-members:
    :exclude-members: __init__, getGlobalToLocalMap, setBufferOrdering

    .. method:: __init__(self: marius._storage.PartitionBufferStorage, filename: str, dim0_size: int, dim1_size: int, options: marius._config.PartitionBufferOptions) -> None

    .. method:: __init__(self: marius._storage.PartitionBufferStorage, filename: str, data: torch.Tensor, options: marius._config.PartitionBufferOptions) -> None
    
    .. method:: __init__(self: marius._storage.PartitionBufferStorage, filename: str, options: marius._config.PartitionBufferOptions) -> None
    
    .. method:: getGlobalToLocalMap(self: marius._storage.PartitionBufferStorage, get_current: bool = True) -> torch.Tensor

    .. method:: setBufferOrdering(self: marius._storage.PartitionBufferStorage, buffer_states: List[torch.Tensor]) -> None

.. autoclass:: marius.storage.InMemory
    :members:
    :undoc-members:
    :exclude-members: __init__

    .. method:: __init__(self: marius._storage.InMemory, filename: str, shape: List[int], dtype: torch.dtype, device: torch.device) -> None

    .. method:: __init__(self: marius._storage.InMemory, filename: str, data: torch.Tensor, device: torch.device) -> None
    
    .. method:: __init__(self: marius._storage.InMemory, filename: str, dtype: torch.dtype) -> None
