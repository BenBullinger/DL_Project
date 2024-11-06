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
    parser.add_argument('--model', type=str.lower, default='gin', choices=["gin", "gt"], help="Selects the model.")
    parser.add_argument('--epochs', type=int, default=100, help="Specifies the number of training epochs")
    parser.add_argument('--data', type=str.upper, default='proteins', choices=["collab","enzymes","proteins","reddit"])
    
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
