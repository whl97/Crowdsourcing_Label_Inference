import numpy as np
import torch

def get_initial_features(wkr_tsk_adj):
    worker_features = torch.from_numpy(wkr_tsk_adj).float()
    task_feature = worker_features.transpose(0, 1)
    return worker_features, task_feature


def extract_feature(mv_results, wkr_tsk_adj, feat_dim):
    mv_predict_labels = np.argmax(mv_results, axis=1)
    num_wkr, num_tsk = wkr_tsk_adj.shape

    ability = np.zeros((num_wkr, feat_dim))
    difficulty = np.zeros((num_tsk, feat_dim))

    for i in range(num_wkr):
        tmp = np.sum(wkr_tsk_adj[i] == -1)
        correct_num = np.sum(np.equal(wkr_tsk_adj[i], mv_predict_labels))
        ability[i, 0] = correct_num / (num_tsk - tmp)

    for j in range(num_tsk):
        tmp = np.sum(wkr_tsk_adj[:, j] == -1)
        correct_num = np.sum(wkr_tsk_adj[:, j] == mv_predict_labels[j])
        difficulty[j, 0] = correct_num / (num_wkr - tmp)

    for i in range(1, feat_dim):
        ability[:, i] = ability[:, 0]
    for i in range(1, feat_dim):
        difficulty[:, i] = difficulty[:, 0]
    feature = np.concatenate((ability, difficulty))
    return torch.from_numpy(feature).unsqueeze(1).float()

