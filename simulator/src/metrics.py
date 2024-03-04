import numpy as np

class MetricTracker:

    def __init__(self):
        self.metrics = {}
    
    def record_metric(self, name, value):
        if name not in self.metrics:
            self.metrics[name] = [0.0, 0]
        self.metrics[name][0] += value
        self.metrics[name][1] += 1
    
    def get_metrics(self):
        metrics_avg = {}
        for metric_name, metric_values in self.metrics.items():
            metrics_avg[metric_name] = (1.0 * metric_values[0])/metric_values[1]

        return metrics_avg