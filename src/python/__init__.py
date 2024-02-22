# isort: skip_file
import os
import sys

# import torch # this import here causes a GIL error
only_python = os.environ.get("MARIUS_NO_BINDINGS", None)

if not only_python:
    try:
        # import torch  # noqa F401

        # load main modules
        from . import _config as config  # RW: import first due to marius/torch omp linking
        import torch  # noqa F401

        # from . import _config as config
        from . import _data as data
        from . import _manager as manager
        from . import _nn as nn
        from . import _pipeline as pipeline
        from . import _report as report
        from . import _storage as storage

        # load submodules
        from ._data import samplers as samplers
        from ._nn import decoders as decoders
        from ._nn import encoders as encoders
        from ._nn import layers as layers
        from ._nn.decoders import edge as edge
        from ._nn.decoders import node as node

        sys.modules[f"{__name__}.config"] = config
        sys.modules[f"{__name__}.data"] = data
        sys.modules[f"{__name__}.data.samplers"] = samplers
        sys.modules[f"{__name__}.manager"] = manager
        sys.modules[f"{__name__}.nn"] = nn
        sys.modules[f"{__name__}.nn.encoders"] = encoders
        sys.modules[f"{__name__}.nn.decoders"] = decoders
        sys.modules[f"{__name__}.nn.decoders.edge"] = edge
        sys.modules[f"{__name__}.nn.decoders.node"] = node
        sys.modules[f"{__name__}.nn.layers"] = layers
        sys.modules[f"{__name__}.pipeline"] = pipeline
        sys.modules[f"{__name__}.report"] = report
        sys.modules[f"{__name__}.storage"] = storage

        __all__ = ["config", "data", "manager", "nn", "pipeline", "report", "storage"]

    except ModuleNotFoundError:
        print("Bindings not installed")
