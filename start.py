"""
Designed to run a single experiment that trains and evaluates a single settings.
Later on we can change it so that it can run multiple settings.
"""

import argparse
import json

from src.train_and_eval import train_and_eval
from src.utils.config import command_line_parser

if __name__ == "__main__":
    args = command_line_parser()
    
    train_and_eval(args)