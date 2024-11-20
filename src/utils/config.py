import argparse
import json
import os

def command_line_parser():
    parser = argparse.ArgumentParser(
        add_help=True,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('-c','--config', type=str, default=None, help="This option lets you use a json config instead of passing all the arguments to the terminal")
    parser.add_argument('--wandb', action='store_true', help="If passed, enables verbose output.")
    parser.add_argument('-device', type=int, default=0, help="Assign GPU slot to the task (use -1 for cpu)")
    parser.add_argument('-v-','--verbose', action='store_true', help="If passed, enables verbose output.")
    parser.add_argument('--seed', type=int, default=0, help="sets the seed")
    parser.add_argument('--model', type=str.lower, default='gin', choices=["gin", "gt"], help="Selects the model.")
    parser.add_argument('--readout', type=str.lower, default='gin', choices=["add", "min", "max"], help="Selects the aggregation method for graph readout. Ignored for node-prediction task")
    parser.add_argument('--batch_size', type=int, default=32, help="Batch size")
    parser.add_argument('--hidden_channels', type=int, default=16, help="Dimension of the hidden representaion")
    parser.add_argument('--epochs', type=int, default=100, help="Specifies the number of training epochs")
    parser.add_argument('--data', type=str.upper, default='proteins', help="Defines the dataset")
    parser.add_argument('--ignore_GNNBenchmark_original_split', action='store_true', help="Ignores the conventional split of GNNBenchmark dataset.")
    parser.add_argument('--dropout', type=float, default=0, help="Probability of dropping out parameters")
    parser.add_argument('--laplacePE', type=int, default=0, help="Specifies the number of Laplacian eigenvectors to add as positional encodings to each graph node.")
    parser.add_argument('--init_nodefeatures_dim', type=int, default=8, help="Dimension of initialized node features. (If the dataset has none)")
    parser.add_argument('--init_nodefeatures_strategy', type=str, default="ones", choices=["random", "zeros", "ones"], help="Strategy to initialize node features (if the dataset has none): 'random' values, all 'zeros', or all 'ones'.")
    parser.add_argument('--sample_transform', action='store_true', help="If passed, adds the sample transform to check the preprocess example.")
    
    args = parser.parse_args()

    if args.config is not None:
        with open(args.config, 'r') as f:
            conf = json.load(f)
            for key in conf.keys():
                setattr(args, key, conf[key])

    args.device = "cpu" if args.device == -1 else args.device

    with open("wandb.key", "r") as file:
        wandb_api_key = file.read().strip()

    print(wandb_api_key) 

    args.wandb = True if args.wandb and wandb_api_key is not None else False
    return args
