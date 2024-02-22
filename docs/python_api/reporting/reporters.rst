
Reporter
=======================================

.. autoclass:: marius.report.Reporter
    :members:
    :undoc-members:
    :special-members: __init__

.. autoclass:: marius.report.LinkPredictionReporter
    :members:
    :undoc-members:
    :special-members: __init__
    :exclude-members: add_result, compute_ranks

    .. method:: add_result(self: marius._report.LinkPredictionReporter, pos_scores: torch.Tensor, neg_scores: torch.Tensor, edges: torch.Tensor = None) -> None
    
    .. method:: compute_ranks(self: marius._report.LinkPredictionReporter, pos_scores: torch.Tensor, neg_scores: torch.Tensor) -> torch.Tensor

.. autoclass:: marius.report.NodeClassificationReporter
    :members:
    :undoc-members:
    :special-members: __init__
    :exclude-members: add_result

    .. method:: add_result(self: marius._report.NodeClassificationReporter, y_true: torch.Tensor, y_pred: torch.Tensor, node_ids: torch.Tensor = None) -> None

.. autoclass:: marius.report.ProgressReporter
    :members:
    :undoc-members:
    :special-members: __init__
