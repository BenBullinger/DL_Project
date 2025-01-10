"""
Designed to run a single experiment that trains and evaluates a single settings.
Later on we can change it so that it can run multiple settings.
"""

import argparse
import json

from src.train_and_eval import train_and_eval
from src.utils.config import command_line_parser
import torch
from src.train_and_eval import benchmark_model, plot_benchmark_results

if __name__ == "__main__":
    args = command_line_parser()
    
    # train_and_eval(args)
    node_sizes = torch.linspace(1500, 10000, 20, dtype=int)
    times = benchmark_model(args, node_sizes=node_sizes, edge_prob=0.1)
    plot_benchmark_results(node_sizes[2:], times[2:])