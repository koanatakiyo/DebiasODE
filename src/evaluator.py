# coding: utf-8

"""
Author: Du Mingzhe (mingzhe@nus.edu.sg)
Date: 27/05/2024
Description: Bias Evaluator
"""

import wandb
import argparse
import json
import os
# from tqdm import tqdm
# from typing import List, Dict 
from datasets import load_dataset
# import re


import pathlib

def main():

    parser = argparse.ArgumentParser(description="Runs BBQ and stereoset evaluator.") 
    parser.add_argument('--benchmark', type=str, choices=['bbq', 'stereoset'])
    parser.add_argument('--test', type=str, default='disambig', choices=['disambig', 'ambig'])
    parser.add_argument('--category', type=str)
    parser.add_argument('--model_name', type=str)
    parser.add_argument('--method', type=str, default='vanilla', 
                        choices=['vanilla', 'self_explanation', 'self_reflection', 'proposal_and_vote'])
    parser.add_argument('--precentage', type=int, default=100)
    parser.add_argument('--cuda', type=str)
    parser.add_argument('--is_sg', type=str,  default="False")
    parser.add_argument('--eval_data_path', type=str, default=None)
    args = parser.parse_args()


    print(pathlib.Path(__file__).parent.resolve())

    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda
    print(f"CUDA_VISIBLE_DEVICES set to: {os.environ['CUDA_VISIBLE_DEVICES']}")


    from bbq_eval import BBQ_Evaluator
    from stereo_eval import StereoSet_Evaluator
    from caller import HF_Caller, OpenAI_Caller, Agent


    abbv_model_name = args.model_name
    if "mistralai/Mistral-7B-Instruct-v0.3" in args.model_name:
        abbv_model_name = "Mistral"
    elif "microsoft/Phi-3-mini-4k-instruct" in args.model_name:
        abbv_model_name = "phi3"
    elif "Meta-Llama-3-8B" in args.model_name:
        abbv_model_name = "llama3"


    wandb.init(project="bias_testing", name=f"sg_{args.is_sg}_test_{args.benchmark}_{args.category}_{abbv_model_name}", reinit=True)
    wandb.config.update(args)
    print(wandb.config)

#load_dataset("McGill-NLP/stereoset", 'intersentence')['validation']


    if wandb.config['benchmark'] == 'bbq':
        evaluator = BBQ_Evaluator(args.model_name)
        result = evaluator.evaluate(category=args.category, test=args.test, precentage=args.precentage, method=args.method)
    elif wandb.config['benchmark'] == 'stereoset':

        if args.is_sg == "True":
            with open("data/stereoset/test_intersentence_sg.json", "r") as jsonfile:
                stereo_dataset = json.load(jsonfile)
                stereo_dataset = stereo_dataset['data']['intersentence']

        else:
            # stereo_dataset = load_dataset("McGill-NLP/stereoset", 'intersentence')['validation']
            with open("data/stereoset/test.json", "r") as jsonfile:
                stereo_dataset = json.load(jsonfile)
                stereo_dataset = stereo_dataset['data']['intersentence']

        evaluator = StereoSet_Evaluator(args.model_name)
        result = evaluator.evaluate(category=args.category, dataset=stereo_dataset)
    else:
        raise NotImplementedError(f"{wandb.config['benchmark']} is unknown.")

    wandb.log({"result": result})
    print(result)
    # wandb.finish()

if __name__ == "__main__":

    main()