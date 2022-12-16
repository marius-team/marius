"""
See https://docs.dgl.ai/tutorials/blitz/3_message_passing.html
"""


class BasicLayer(mpi.Module):
    def __init__(self, input_dim: int, output_dim: int):
        super(mpi.Module, self).__init__(input_dim, output_dim)
        self.linear = mpi.Linear(input_dim * 2, output_dim)
        self.reset_parameters()

    def reset_parameters(self):
        self.linear.reset_parameters()

    def forward(self, graph: mpi.DENSEGraph, h: mpi.Tensor) -> mpi.Tensor:
        with graph.local_scope():
            graph.ndata["h"] = h
            graph.update_all(message_func=mpi.copy_u("h", "m"), reduce_func=mpi.mean("m", "h_N"))
            h_N = graph.ndata["h_N"]
            h_total = mpi.cat(h, h_N, dim=1)
            return self.linear(h_total)
