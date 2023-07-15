import os
import pdb
import time
import random
import torch
import numpy as np
import scipy.io as sio
import scipy.sparse as ss
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize
import scipy.io as scio


def load_data(args):
    data = sio.loadmat(args.path + args.dataset + '.mat')
    features = data['X']
    feature_list = []
    adj_list = []
    adj_hat_list = [] # A^
    adj_wave_list = [] # A~

    labels = data['Y'].flatten()
    labels = labels - min(set(labels))
    idx_labeled, idx_unlabeled = generate_partition(labels, args.ratio, args.sf_seed)
    labels = torch.from_numpy(labels).long()

    for i in range(features.shape[1]):
        # print("Loading the data of" + str(i) + "th view")
        features[0][i] = normalize(features[0][i])
        feature = features[0][i]
        # print(np.linalg.norm(feature[0]))
        if ss.isspmatrix_csr(feature):
            feature = feature.todense()
            print("sparse")
        direction_judge = './adj_matrix/' + args.dataset + '/' + 'v' + str(i) + '_knn' + str(args.knns) + '_adj.npz'
        if os.path.exists(direction_judge):
            print("Loading the adjacency matrix of " + str(i) + "th view of " + args.dataset)
            # adj = torch.from_numpy(ss.load_npz(direction_judge).todense()).float().to(args.device)
            adj = ss.load_npz(direction_judge).todense()
        # construct the furthest adj
        direction_judge1 = './adj_matrix_f/' + args.dataset + '/' + 'v' + str(i) + '_knn' + str(args.knns) + '_adj.npz'
        if os.path.exists(direction_judge1):
            print("Loading the furthest adjacency matrix of " + str(i) + "th view of " + args.dataset)
            adj_f = ss.load_npz(direction_judge1).todense()

        feature = torch.from_numpy(feature).float().to(args.device)
        feature_list.append(feature)
        adj_hat_list.append(torch.from_numpy(construct_adj_hat(adj).todense()).float().to(args.device))
        adj_wave_list.append(torch.from_numpy(construct_adj_wave(adj_f).todense()).float().to(args.device))

    return feature_list, adj_hat_list, adj_wave_list, labels, idx_labeled, idx_unlabeled


def construct_adj_hat(adj):
    """
    :param adj: original Laplacian matrix  <class 'scipy.sparse.csr.csr_matrix'>
    :return:
    """
    adj = ss.coo_matrix(adj)
    adj_ = ss.eye(adj.shape[0]) + adj
    # adj_ = adj
    rowsum = np.array(adj_.sum(1))  # <class 'numpy.ndarray'> (n_samples, 1)
    print("mean_degree:", rowsum.mean())
    degree_mat_inv_sqrt = ss.diags(np.power(rowsum, -0.5).flatten())  # degree matrix
    # <class 'scipy.sparse.coo.coo_matrix'>  (n_samples, n_samples)
    adj_hat = adj_.dot(degree_mat_inv_sqrt).transpose().dot(degree_mat_inv_sqrt)
    return adj_hat


def construct_adj_wave(adj):
    """
    :param adj: original Laplacian matrix  <class 'scipy.sparse.csr.csr_matrix'>
    :return:
    """
    adj = ss.coo_matrix(adj)
    # adj_ = 2 * ss.eye(adj.shape[0]) - adj
    adj_ = ss.eye(adj.shape[0]) - adj
    rowsum = np.array(abs(adj_).sum(1))  # <class 'numpy.ndarray'> (n_samples, 1)
    print("mean_degree:", rowsum.mean())
    degree_mat_inv_sqrt = ss.diags(np.power(rowsum, -0.5).flatten())
    # <class 'scipy.sparse.coo.coo_matrix'>  (n_samples, n_samples)
    adj_wave = adj_.dot(degree_mat_inv_sqrt).transpose().dot(degree_mat_inv_sqrt).tocoo()
    return adj_wave


def construct_sparse_float_tensor(np_matrix):
    """
        construct a sparse float tensor according a numpy matrix
    :param np_matrix: <class 'numpy.ndarray'>
    :return: torch.sparse.FloatTensor
    """
    sp_matrix = ss.csc_matrix(np_matrix)
    three_tuple = sparse_to_tuple(sp_matrix)
    sparse_tensor = torch.sparse.FloatTensor(torch.LongTensor(three_tuple[0].T),
                                             torch.FloatTensor(three_tuple[1]),
                                             torch.Size(three_tuple[2]))
    return sparse_tensor


def sparse_to_tuple(sparse_mx):
    if not ss.isspmatrix_coo(sparse_mx):
        sparse_mx = sparse_mx.tocoo()

    # sparse_mx.row/sparse_mx.col  <class 'numpy.ndarray'> [   0    0    0 ... 2687 2694 2706]
    coords = np.vstack((sparse_mx.row, sparse_mx.col)).transpose()  # <class 'numpy.ndarray'> (n_edges, 2)
    values = sparse_mx.data  # <class 'numpy.ndarray'> (n_edges,) [1 1 1 ... 1 1 1]
    shape = sparse_mx.shape  # <class 'tuple'>  (n_samples, n_samples)
    return coords, values, shape


def generate_partition(labels, ratio, seed):
    each_class_num = count_each_class_num(labels)
    labeled_each_class_num = {} ## number of labeled samples for each class
    total_num = round(ratio * len(labels))
    for label in each_class_num.keys():
        labeled_each_class_num[label] = max(round(each_class_num[label] * ratio), 1) # min is 1

    # index of labeled and unlabeled samples
    p_labeled = []
    p_unlabeled = []
    index = [i for i in range(len(labels))]
    # print(index)
    if seed >= 0:
        random.seed(seed)
        random.shuffle(index)
    labels = labels[index]
    for idx, label in enumerate(labels):
        if (labeled_each_class_num[label] > 0):
            labeled_each_class_num[label] -= 1
            p_labeled.append(index[idx])
            total_num -= 1
        else:
            p_unlabeled.append(index[idx])
    return p_labeled, p_unlabeled


def count_each_class_num(labels):
    '''
        Count the number of samples in each class
    '''
    count_dict = {}
    for label in labels:
        if label in count_dict.keys():
            count_dict[label] += 1
        else:
            count_dict[label] = 1
    return count_dict
