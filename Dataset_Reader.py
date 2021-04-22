from utils import *

datasets_dir = "all_datasets/"
dataset_namelist = ['bluebird', 
                    'flowers', 
                    'web', 
                    'dog', 
                    'rte', 
                    'SP', 
                    'SP_amt',
                    'ZenCrowd_all', 
                    'ZenCrowd_in',
                    'ZenCrowd_us', 
                    'face', 
                    'product', 
                    'sentiment']


def read_dataset(dataset_id):
    dataset_name = dataset_namelist[dataset_id - 1]
    dataset_path = datasets_dir + dataset_name + '.npz'
    data_all = np.load(dataset_path)
    wkr_tsk_adj = data_all['worker_task_matrix']
    num_wkr = data_all['n_worker']
    num_tsk = data_all['n_task']
    num_rels = data_all['n_categories']
    true_labels = data_all['gt_labels']
    name = str(data_all['dataset_name'])
    return wkr_tsk_adj, Graph2Edgelist(wkr_tsk_adj), num_wkr + num_tsk, num_rels, \
           num_wkr, num_tsk, true_labels, name, get_mv_result(wkr_tsk_adj, num_rels)





