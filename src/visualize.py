import matplotlib.pyplot as plt

def plot_metrics(metrics_dict, metric_name, colors):
    models = list(metrics_dict.keys())
    values = [metrics[metric_name] for metrics in metrics_dict.values()]
    
    plt.bar(models, values, color=colors)
    plt.ylim(bottom=max(0, min(values) - 0.1))
    plt.title(metric_name.replace("_", " ").title())
    plt.grid(True)
    plt.show()
