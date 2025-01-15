"""
Designed to run a single experiment that trains and evaluates a single settings.
Later on we can change it so that it can run multiple settings.
"""

import argparse
import json

from src.train_and_eval import benchmark_model, plot_benchmark_results
from src.utils.config import command_line_parser
import torch

if __name__ == "__main__":
    args = command_line_parser()
    
    node_sizes = torch.linspace(1000, 10000, 10, dtype=int)
    times = benchmark_model(args, node_sizes=node_sizes, edge_prob=0.1)
    print(f'Zipped list of (node_sizes, seconds of runtime): {list(zip(node_sizes.tolist(), times))}')
    # plot_benchmark_results(node_sizes[2:], times[2:])