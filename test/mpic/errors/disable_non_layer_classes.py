class ErroneousClass:
    def __init__(self, input_dim: int, output_dim: int):
        def init():
            pass

        init()

    def reset_parameters(self):
        pass

    def forward(self, graph: mpi.DENSEGraph, h: torch.Tensor) -> torch.Tensor:
        return h
