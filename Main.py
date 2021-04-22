import argparse
import torch.nn as nn
from Model import PredictModel
from utils import *
import Feature_Initializer
import Dataset_Reader
import Train_data_Generator
import torch.nn.functional as F
import time
from sklearn.model_selection import KFold
import numpy as np
from tqdm import tqdm, trange

def main(args):

    wkr_tsk_adj, data, num_nodes, num_rels, num_wkr, num_tsk, true_labels, name, mv_results = Dataset_Reader.read_dataset(
        args.dataset_id)

    print("="*100)
    print("Model type:", args.model_type)
    if "mid" in args.model_type:
        print("Num of heads: ", args.num_heads)
    print("Dataset name:", name)
    print("Learning rate", args.lr)
    print("training proportion:", args.train_proportion)
    print("="*100)

    features = Feature_Initializer.extract_feature(mv_results, wkr_tsk_adj, args.feat_dim)
    use_cuda = args.gpu >= 0 and torch.cuda.is_available()
    if use_cuda:
        torch.cuda.set_device(args.gpu)

    transverse_data = data[:, [2, 1, 0]] 
    triplets_before = np.concatenate((data, transverse_data), axis=0) 
    triplets = (data[:, 0], data[:, 1], data[:, 2])  

    true_labels = torch.from_numpy(true_labels.astype(float)).long()  
    data = torch.from_numpy(data.astype(float))

    g = build_graph_from_triplets(num_nodes, triplets) 
    edge_type = torch.tensor(triplets_before[:, 1].astype(int))  
    node_deg = g.out_degrees(g.nodes()).float().view(-1, 1, 1)
    node_id = torch.tensor([i for i in range(0, num_nodes)])
    if use_cuda:
        edge_type, node_deg, node_id = edge_type.cuda(), node_deg.cuda(), node_id.cuda()
    g.edata.update({'type': edge_type})
    g.ndata.update({'deg': node_deg, 'id': node_id})


    train_data = Train_data_Generator.get_cross_val_train_data_from_files(args.dataset_id, args.train_proportion)
    n_splits = len(train_data) 

    all_splits_results = []
    majority_voting_mean_acc = 0

    progress_bar = trange(int(n_splits), desc="Progress bar for a dataset")

    for i in progress_bar:
        train_idx = train_data[i]
        test_idx = list(set(list(range(num_tsk))) - set(train_idx))
        train_idx = torch.tensor(train_idx).long()
        test_idx = torch.tensor(test_idx).long()

        each_epoch_results = []

        model = PredictModel(model_type=args.model_type,
                            g=g,
                            num_nodes=num_nodes,
                            num_wkr=num_wkr,
                            num_tsk=num_tsk,
                            num_rels=num_rels,
                            feat_dim=args.feat_dim,
                            mv_results=mv_results,
                            num_heads = args.num_heads,
                            dropout=args.dropout,
                            use_cuda=use_cuda,
                            reg_param=args.regularization,
                            e_dim=args.e_dim,
                            feat_init=args.feat_init
                            )
        if use_cuda:
            model.cuda()

        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

        max_epoch = args.n_epochs
        best_train_acc = 0
        best_test_acc = 0
        best_total_acc = 0

        pbar_for_one_split = trange(int(max_epoch), desc="Progress for one split", )
        for epoch in pbar_for_one_split:
            model.train()
            if use_cuda:
                features, data, train_idx, true_labels = features.cuda(), data.cuda(), train_idx.cuda(), true_labels.cuda()
            label_loss, edge_loss, loss, predict_label_score = model.get_loss(features, data, train_idx, true_labels)

            predict_labels = torch.argmax(predict_label_score, dim=2).squeeze(1)
            acc_train = torch.sum(predict_labels[train_idx] == true_labels[train_idx].long()).item() / train_idx.shape[0]
            acc_test = torch.sum(predict_labels[test_idx] == true_labels[test_idx].long()).item() / test_idx.shape[0]
            acc_total = torch.sum(predict_labels == true_labels.long()).item() / num_tsk

            each_epoch_results.append(acc_test)
            if acc_train > best_train_acc:
                best_train_acc = acc_train
            if acc_test > best_test_acc:
                best_test_acc = acc_test
            if acc_total > best_total_acc:
                best_total_acc = acc_total
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), args.grad_norm)
            optimizer.step()
            pbar_for_one_split.set_postfix(test_acc=acc_test)

            optimizer.zero_grad()
            epoch += 1

        all_splits_results.append(each_epoch_results)
    majority_voting_mean_acc /= n_splits
    all_splits_results = np.array(all_splits_results)
    test_acc_results = (np.sum(all_splits_results, axis=0) / n_splits)  # mean
    best_test_acc_epoch = np.argmax(test_acc_results, axis=0)
    final_test_acc = test_acc_results[best_test_acc_epoch]
    final_all_splits_test_acc = all_splits_results[:,best_test_acc_epoch]

    print(
        "\ndataset: {:d} | acc: {:.4f}| lr: {:.5f} | train proportion: {:.3f}| max epochs: {:d} | best_acc_epoch: {:d}| seed: {:d}\n".format(
            args.dataset_id,
            final_test_acc,
            args.lr,
            args.train_proportion,
            args.n_epochs,
            best_test_acc_epoch,
            args.seed))
    print(final_test_acc)
    print(final_all_splits_test_acc)

    with open(args.filename, 'a') as f:
        f.write(
            "\ndataset: {:d} | acc: {:.4f}| lr: {:.5f} | train proportion: {:.3f}| max epochs: {:d} | best_acc_epoch: {:d}| seed: {:d}\n".format(
                args.dataset_id,
                final_test_acc,
                args.lr,
                args.train_proportion,
                args.n_epochs,
                best_test_acc_epoch,
                args.seed))
        f.write("{} splits:".format(n_splits))
        for i in range(n_splits):
            f.write(str(final_all_splits_test_acc[i])+"  ")

    torch.save(model, args.model_file)





if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument("-m", "--model-type", type=str, default="mp2_mid_mp2") 
    parser.add_argument("--feat-init", type=str, default="same") 
    parser.add_argument("--num-heads", type=int, default=1) 
    parser.add_argument("-d", "--dataset-id", type=int, default=1)
    parser.add_argument("--train-proportion", type=float, default=0.2)
    parser.add_argument("--feat-dim", type=int, default=30) 
    parser.add_argument("--e-dim", type=int, default=20) 
    parser.add_argument("--seed", type=int, default=10)  
    parser.add_argument("--lr", type=float, default=1e-2)  
    parser.add_argument("--n-epochs", type=int, default=300)
    parser.add_argument("--regularization", type=float, default=0.01)
    parser.add_argument("--grad-norm", type=float, default=1.0)
    parser.add_argument("--dropout", type=float, default=0)
    parser.add_argument("--model-file", type=str)
    parser.add_argument("--filename", type=str)
    parser.add_argument("--gpu", type=int, default=1)
    
    args = parser.parse_args()
    setup_seed(args.seed)
    main(args)



