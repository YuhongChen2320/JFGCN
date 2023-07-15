"""
@Project   : JFGCN
@Time      : 2023/7/15
@Author    : Yuhong Chen
@File      : args.py
"""
import argparse

def parameter_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cpu", help="Device: cuda:num or cpu")
    parser.add_argument("--path", type=str, default="./datasets/", help="Path of datasets")
    parser.add_argument("--dataset", type=str, default="ALOI", help="Name of datasets")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for train-test split. Default is 42.")
    parser.add_argument("--sf_seed", type=int, default=2021, help="Random seed for train-test split. Default is 42.")
    parser.add_argument("--fix_seed", action='store_true', default=True, help="xx")

    parser.add_argument("--n_repeated", type=int, default=1, help="Number of repeated times. Default is 10.")
    parser.add_argument("--knns", type=int, default=10, help="Number of k nearest neighbors")

    parser.add_argument("--lr", type=float, default=0.0030, help="ALOI")
    parser.add_argument("--attentionlist", type=list, default=[7,2], help="xx")
    parser.add_argument("--weight_decay", type=float, default= 5e-4, help="Weight decay")
    parser.add_argument("--ratio", type=float, default=0.1, help="Ratio of labeled samples")
    parser.add_argument("--num_epoch", type=int, default=1000, help="Number of training epochs. Default is 200.")


    args = parser.parse_args()

    return args