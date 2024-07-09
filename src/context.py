
import torch
import os
import re
import wandb
import argparse
from tqdm import tqdm


class sg_context:


    def __init__(self, dataset, cuda) -> None:
        self.dataset = load_dataset(path="Elfsong/BBQ", split=f'{category}[:{precentage}%]')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--benchmark', type=str, choices=['bbq', 'stereoset'])
    parser.add_argument('--test', type=str, default='disambig', choices=['disambig', 'ambig'])
    parser.add_argument('--category', type=str)
    parser.add_argument('--model_name', type=str)
    parser.add_argument('--method', type=str, default='vanilla', 
                        choices=['vanilla', 'self_explanation', 'self_reflection', 'proposal_and_vote'])
    parser.add_argument('--precentage', type=int, default=100)
    parser.add_argument('--cuda', type=str, choices=["0","1","2","3","4","5","6","7"])
    args = parser.parse_args()
