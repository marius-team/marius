class ErroneousLayer(nn.Module):
    def __init__(self, input_dim: int, output_dim: int):
        output_dim, input_dim = input_dim, output_dim

    def reset_parameters(self):
        pass

    def forward(self, graph: mpi.DENSEGraph, h: torch.Tensor) -> torch.Tensor:
        return h
