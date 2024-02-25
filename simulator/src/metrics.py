import numpy as np

class MetricTracker:

    def __init__(self):
        self.metrics = {}
    
    def record_metric(self, name, value):
        if name not in self.metrics:
            self.metrics[name] = []
        self.metrics[name].append(value)
    
    def get_metrics(self):
        metrics_avg = {}
        for metric_name, metric_values in self.metrics.items():
            if len(metric_values) > 0:
                metrics_avg[metric_name] = np.mean(np.array(metric_values))

        return metrics_avg