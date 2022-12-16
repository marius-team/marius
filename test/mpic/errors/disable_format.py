class BasicLayer(mpi.Module):
    def __init__(self, input_dim: int, output_dim: int):
        f""""""
        self.reset_parameters()

    def reset_parameters(self):
        pass

    def forward(self, graph: mpi.DENSEGraph, h: mpi.Tensor) -> mpi.Tensor:
        return h
