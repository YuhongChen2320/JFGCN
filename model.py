"""
@Project   : JFGCN
@Time      : 2023/7/15
@Author    : Yuhong Chen
@File      : model.py
"""
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import sys


class InstanceLoss(nn.Module):
    def __init__(self, batch_size, temperature, device):
        super(InstanceLoss, self).__init__()
        self.batch_size = batch_size
        self.temperature = temperature
        self.device = device

        self.mask = self.mask_correlated_samples(batch_size)
        self.criterion = nn.CrossEntropyLoss(reduction="sum")

    def mask_correlated_samples(self, batch_size):
        N = 2 * batch_size
        mask = torch.ones((N, N))
        mask = mask.fill_diagonal_(0)
        for i in range(batch_size):
            mask[i, batch_size + i] = 0
            mask[batch_size + i, i] = 0
        mask = mask.bool()
        return mask

    def forward(self, z_i, z_j):
        N = 2 * self.batch_size
        z = torch.cat((z_i, z_j), dim=0)

        sim = torch.matmul(z, z.T) / self.temperature
        sim_i_j = torch.diag(sim, self.batch_size)
        sim_j_i = torch.diag(sim, -self.batch_size)

        positive_samples = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1)
        negative_samples = sim[self.mask].reshape(N, -1)

        labels = torch.zeros(N).to(positive_samples.device).long()
        logits = torch.cat((positive_samples, negative_samples), dim=1)
        loss = self.criterion(logits, labels)
        loss /= N

        return loss





class FAGC(nn.Module):
    def __init__(self, input_dim, output_dim, n, **kwargs):
        super(FAGC, self).__init__(**kwargs)
        self.weight = glorot_init(input_dim, output_dim)
        # self.alpha = nn.Parameter(torch.ones(2, n, n))

    def forward(self, inputs, adj):
        # alpha = F.softmax(self.alpha, dim=0)
        x = inputs
        x = torch.mm(x, self.weight)
        x = torch.mm(adj, x)
        return x


def glorot_init(input_dim, output_dim):
    init_range = np.sqrt(6.0/(input_dim + output_dim))
    initial = torch.rand(input_dim, output_dim)*2*init_range - init_range
    return nn.Parameter(initial)


class FAGCN(nn.Module):
    def __init__(self, hidden_dims, num_view, n,attentionlist):
        super(FAGCN, self).__init__()
        self.gc = nn.ModuleList()
        for i in range(len(hidden_dims)-1):
            self.gc.append(FAGC(hidden_dims[i], hidden_dims[i+1], n))
        self.a = nn.Parameter(torch.ones(num_view))
        self.b = nn.Parameter(torch.ones(num_view))
        self.c = nn.Parameter(torch.Tensor([7,5]), requires_grad=True)
        self.num_view = num_view

    def forward(self, x, adj_hat_list, adj_wave_list):
        a = F.softmax(self.a, dim=0)
        b = F.softmax(self.b, dim=0)
        c = F.softmax(self.c, dim=0)
        adj_hat = sum([w * e for e, w in zip(a, adj_hat_list)])
        adj_wave = sum([w * e for e, w in zip(b, adj_wave_list)])
        emb1 = x
        emb2 = x
        for gc in self.gc[:-1]:
            emb1 = F.relu(gc(emb1, adj_hat))
            emb2 = F.relu(gc(emb2, adj_wave))
        emb1 = self.gc[-1](emb1, adj_hat)
        emb2 = self.gc[-1](emb2, adj_wave)
        return c[0] * emb1 + c[1] * emb2, emb1, emb2,c[0],c[1]


class DeepMvNMF(nn.Module):
    def __init__(self, input_dims, en_hidden_dims, num_views, device):
        super(DeepMvNMF, self).__init__()
        self.encoder = nn.ModuleList()
        self.mv_decoder = nn.ModuleList()
        self.device = device
        for i in range(len(en_hidden_dims)-1):
            self.encoder.append(nn.Linear(en_hidden_dims[i], en_hidden_dims[i+1]))
        for i in range(num_views):
            decoder = nn.ModuleList()
            de_hidden_dims = [input_dims[i]]
            for k in range(1, len(en_hidden_dims)):
                de_hidden_dims.insert(0, en_hidden_dims[k])
            # print(de_hidden_dims)
            for j in range(len(de_hidden_dims)-1):
                decoder.append(nn.Linear(de_hidden_dims[j], de_hidden_dims[j+1]))
            self.mv_decoder.append(decoder)
        # print(self.mv_decoder)

    def forward(self, input):
        z = input
        for layer in self.encoder:
            z = F.relu(layer(z))
        x_hat_list = []
        for de in self.mv_decoder:
            x_hat = z
            for layer in de:
                x_hat = F.relu(layer(x_hat))
            x_hat_list.append(x_hat)
        return z, x_hat_list
