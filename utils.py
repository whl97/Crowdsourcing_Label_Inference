import numpy as np
import yaml
import dgl
import random
import torch
from collections import Counter
import os


def setup_seed(seed=20):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def Graph2Edgelist(Graph):
    workers_num = Graph.shape[0]
    tasks_num = Graph.shape[1]
    edge_list = []
    for worker in range(workers_num):
        for task in range(tasks_num):
            if Graph[worker][task] != -1:
                edge_list.append([worker, Graph[worker][task], workers_num + task])
    return np.array(edge_list)


def get_mv_result(graph, num_rels):
    num_wkr = graph.shape[0]
    num_tsk = graph.shape[1]

    majority_voting_result = np.zeros((num_tsk, num_rels)) - 1  

    for i in range(num_tsk):
        _tsk_labeling = list(graph[:, i]) 
        label_votes_dict = Counter(_tsk_labeling) 
        if label_votes_dict[-1] == num_wkr: 
            continue
        num_crowd_labels = num_wkr - label_votes_dict[-1]
        for l in range(num_rels):
            majority_voting_result[i, l] = label_votes_dict[l] / num_crowd_labels
    return majority_voting_result


def build_graph_from_triplets(num_nodes, triplets):
    g = dgl.DGLGraph()
    g.add_nodes(num_nodes)
    src, rel, dst = triplets
    src, dst = np.concatenate((src, dst)), np.concatenate((dst, src))
    g.add_edges(src, dst)
    return g



