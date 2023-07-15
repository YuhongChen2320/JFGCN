"""
@Project   : JFGCN
@Time      : 2023/7/15
@Author    : Yuhong Chen
@File      : train.py.py
"""
import os
import random
import time
import warnings

import matplotlib.pyplot as plt
import numpy as np
import scipy.io as scio
import torch
import torch.nn.functional as F
from sklearn import manifold
from tqdm import tqdm

from Dataloader import load_data
from args import parameter_parser
from model import FAGCN, DeepMvNMF
from utils import tab_printer, get_evaluation_results, norm_2


def train(args, device):
    feature_list, adj_hat_list, adj_wave_list, labels, idx_labeled, idx_unlabeled = load_data(args)

    num_classes = len(np.unique(labels))
    labels = labels.to(device)
    N = feature_list[0].shape[0]
    num_view = len(feature_list)
    input_dims = []
    for i in range(num_view): # multiview data { data includes features and ... }
        input_dims.append(feature_list[i].shape[1])

    en_hidden_dims = [N, 2048, 1024]
    DMF_model = DeepMvNMF(input_dims, en_hidden_dims, num_view, device).to(device)
    optimizer_DMF = torch.optim.Adam(DMF_model.parameters(), lr=1e-3, weight_decay=5e-5)

    hidden_dims = [1024, 32, num_classes]
    GCN_model = FAGCN(hidden_dims, num_view, N,args.attentionlist).to(device)
    optimizer_GCN = torch.optim.Adam(GCN_model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    identity = torch.eye(feature_list[0].shape[0]).to(device)


    loss_function1 = torch.nn.NLLLoss()

    Loss_list = []
    ACC_list = []
    F1_list = []
    begin_time = time.time()
    with tqdm(total=args.num_epoch, desc="Pretraining") as pbar:
        for epoch in range(args.num_epoch):
            shared_z, x_hat_list = DMF_model(identity)
            loss_DMF = 0.
            for i in range(num_view):
                loss_DMF += norm_2(feature_list[i], x_hat_list[i])
            optimizer_DMF.zero_grad()
            loss_DMF.backward()
            optimizer_DMF.step()
            pbar.set_postfix({'Loss': '{:.6f}'.format(loss_DMF.item())})
            pbar.update(1)
    with tqdm(total=args.num_epoch, desc="Training") as pbar:
        for epoch in range(args.num_epoch):
            shared_z, x_hat_list = DMF_model(identity)
            loss_DMF = 0.
            for i in range(num_view):
                loss_DMF += norm_2(feature_list[i], x_hat_list[i])
            optimizer_DMF.zero_grad()
            loss_DMF.backward()
            optimizer_DMF.step()

            GCN_input = shared_z.detach()
            GCN_model.train()
            z,_,_,c0,c1 = GCN_model(GCN_input, adj_hat_list, adj_wave_list)
            output = F.log_softmax(z, dim=1)
            output1 = F.softmax(z,dim=1)
            optimizer_GCN.zero_grad()
            loss_GCN = loss_function1(output[idx_labeled], labels[idx_labeled])
            loss_GCN.backward()
            optimizer_GCN.step()
            with torch.no_grad():
                GCN_model.eval()
                # output, _, _ = model(feature_list, lp_list, args.Lambda, args.ortho)
                pred_labels = torch.argmax(output, 1).cpu().detach().numpy()
                ACC, _, _, F1,AUC = get_evaluation_results(labels.cpu().detach().numpy()[idx_unlabeled], pred_labels[idx_unlabeled],output1.cpu().detach().numpy()[idx_unlabeled])
                pbar.set_postfix({'Loss': '{:.6f}'.format((loss_GCN + loss_DMF).item()),
                                  'ACC': '{:.2f}'.format(ACC * 100), 'F1': '{:.2f}'.format(F1 * 100)})
                pbar.update(1)
                Loss_list.append(float((loss_GCN + loss_DMF).item()))
                ACC_list.append(ACC)
                F1_list.append(F1)
    cost_time = time.time() - begin_time
    GCN_model.eval()
    z,_,_,c0,c1 = GCN_model(GCN_input, adj_hat_list, adj_wave_list)
    print("Evaluating the model")
    pred_labels = torch.argmax(z, 1).cpu().detach().numpy()

    ACC, P, R, F1, AUC = get_evaluation_results(labels.cpu().detach().numpy()[idx_unlabeled], pred_labels[idx_unlabeled],output1.cpu().detach().numpy()[idx_unlabeled])
    print("------------------------")
    print("ACC:   {:.2f}".format(ACC * 100))
    print("F1 :   {:.2f}".format(F1 * 100))
    print("------------------------")

    return ACC, P, R, F1,AUC, cost_time, Loss_list, ACC_list, F1_list,c0,c1



if __name__ == '__main__':
    warnings.filterwarnings('ignore')
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    args = parameter_parser()
    device = torch.device('cpu' if args.device == 'cpu' else 'cuda:' + args.device)
    args.device = device
    if args.fix_seed:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
    tab_printer(args)

    all_ACC = []
    all_P = []
    all_R = []
    all_F1 = []
    all_AUC = []
    all_TIME = []
    for i in range(args.n_repeated):
        torch.cuda.empty_cache()
        ACC, P, R, F1,AUC, Time, Loss_list, ACC_list, F1_list,c0,c1 = train(args, device)
        all_ACC.append(ACC)
        all_P.append(P)
        all_R.append(R)
        all_F1.append(F1)
        all_AUC.append(AUC)
        all_TIME.append(Time)

    print("-----------------------")
    print("ACC: {:.2f} ({:.2f})".format(np.mean(all_ACC) * 100, np.std(all_ACC) * 100))
    print("P  : {:.2f} ({:.2f})".format(np.mean(all_P) * 100, np.std(all_P) * 100))
    print("R  : {:.2f} ({:.2f})".format(np.mean(all_R) * 100, np.std(all_R) * 100))
    print("F1 : {:.2f} ({:.2f})".format(np.mean(all_F1) * 100, np.std(all_F1) * 100))
    print("AUC : {:.2f} ({:.2f})".format(np.mean(all_AUC) * 100, np.std(all_AUC) * 100))
    print("-----------------------")

