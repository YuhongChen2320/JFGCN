"""
@Project   : JFGCN
@Time      : 2023/7/15
@Author    : Yuhong Chen
@File      : utils.py
"""
import torch
from texttable import Texttable
from sklearn import metrics
from sklearn.metrics import roc_auc_score

def tab_printer(args):
    args = vars(args)
    keys = sorted(args.keys())
    t = Texttable()
    t.add_rows([["Parameter", "Value"]] + [[k.replace("_", " ").capitalize(), args[k]] for k in keys])
    print(t.draw())


def get_evaluation_results(labels_true, labels_pred,output1):
    ACC = metrics.accuracy_score(labels_true, labels_pred)
    P = metrics.precision_score(labels_true, labels_pred, average='macro')
    R = metrics.recall_score(labels_true, labels_pred, average='macro')
    F1 = metrics.f1_score(labels_true, labels_pred, average='macro')
    AUC = roc_auc_score(labels_true, output1, multi_class='ovr')
    return ACC, P, R, F1, AUC


def norm_2(x, y):
    return 0.5 * (torch.norm(x-y) ** 2)