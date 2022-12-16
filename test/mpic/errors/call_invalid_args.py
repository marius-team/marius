class ErroneousLayer(mpi.Module):
    def __init__(self, input_dim: int, output_dim: int):
        self.reset_parameters(0)

    def reset_parameters(self):
        pass

    def forward(self, graph: mpi.DENSEGraph, h: torch.Tensor) -> torch.Tensor:
        return h
