import torch
import torch.nn as nn
from Encoder import Encoder

class PredictModel(nn.Module):
    def __init__(self, model_type, g, num_nodes, num_wkr, num_tsk, num_rels, feat_dim, mv_results,num_heads,
                 dropout=0, use_cuda=False, reg_param=0, e_dim=20, feat_init=None):
        super(PredictModel, self).__init__()
        self.num_nodes = num_nodes
        self.num_wkr = num_wkr
        self.num_tsk = num_tsk
        self.feat_dim = feat_dim
        self.num_rels = num_rels
        self.mv_results = mv_results
        self.g = g
        self.model_type = model_type
        self.num_heads = num_heads
        self.feat_init = feat_init 
        self.fc_trans_wkr = nn.Linear(num_tsk, feat_dim, bias=True)
        self.fc_trans_tsk = nn.Linear(num_wkr, feat_dim, bias=True)

        self.fc_edge_pred = nn.Linear(feat_dim * 2, num_rels, bias=True)
        self.fc_label_pred = nn.Linear(feat_dim, num_rels, bias=True)


        self.encoder = Encoder(model_type=model_type, g=g, num_nodes=num_nodes, num_wkr=num_wkr, num_tsk=num_tsk, feat_dim=feat_dim,
                               num_rels=num_rels, num_heads=num_heads)

    def predict_edge_score(self, ndata_h, triplets):
        wkr = triplets[:, 0]
        tsk = triplets[:, 2]
        wkr_feature = ndata_h[wkr.long()].squeeze(1)  
        tsk_feature = ndata_h[tsk.long()].squeeze(1)  

        edge_predict_score = self.fc_edge_pred(torch.cat((wkr_feature, tsk_feature), dim=1))
        return edge_predict_score

    def predict_label_score(self, ndata_h):
        tsk_node_feature = ndata_h[self.num_wkr:self.num_nodes]
        tsk_label_score = self.fc_label_pred(tsk_node_feature)
        return tsk_label_score

    def regularization_loss(self, embedding):
        return torch.mean(embedding.pow(2))  

    def get_loss(self, same_feat, triplets, train_tsk_id, true_labels):
        if self.feat_init=="same":
            features = same_feat
        elif self.feat_init=="rand":
            features = torch.rand_like(same_feat)  
        ndata_h = self.encoder.forward(features)
        wkr = triplets[:, 0]
        rel = triplets[:, 1]
        tsk = triplets[:, 2]

        loss = nn.CrossEntropyLoss()
        edge_predict_score = self.predict_edge_score(ndata_h, triplets)
        predict_edge_loss = loss(edge_predict_score, rel.squeeze().long())
        label_predict_score = self.predict_label_score(ndata_h)
        predict_label_loss = loss(label_predict_score[train_tsk_id].squeeze(1), true_labels[train_tsk_id])
        loss_sum = predict_label_loss 
        return predict_label_loss, predict_edge_loss, loss_sum, label_predict_score
