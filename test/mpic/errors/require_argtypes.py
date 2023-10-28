class ErroneousLayer(nn.Module):
    def __init__(self, input_dim, output_dim):
        pass

    def reset_parameters(self):
        pass

    def forward(self, graph, h):
        return h
