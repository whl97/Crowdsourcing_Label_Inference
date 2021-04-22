import torch.nn as nn
from Layers import MPNN_1, MPNN_2, middle_layer
import torch
import torch.nn.functional as F


class Encoder(nn.Module):
    def __init__(self, model_type, g, num_nodes, num_wkr, num_tsk, feat_dim, num_rels, num_heads, e_dim=20):
        super(Encoder, self).__init__()
        self.num_nodes = num_nodes
        self.num_wkr = num_wkr
        self.num_tsk = num_tsk
        self.feat_dim = feat_dim
        self.num_rels = num_rels
        self.layers = nn.ModuleList()
        self.g = g
        self.model_type = model_type

        if model_type == "mp1":
            self.layers.append(MPNN_1(self.feat_dim, self.num_rels))
        elif model_type == "mp1_mp1":
            self.layers.append(MPNN_1(self.feat_dim, self.num_rels))
            self.layers.append(MPNN_1(self.feat_dim, self.num_rels))
        elif model_type == "mp1_mid_mp1":
            self.layers.append(MPNN_1(self.feat_dim, self.num_rels))
            self.layers.append(middle_layer(self.feat_dim, self.num_wkr, self.num_tsk, num_heads=num_heads)) 
            self.layers.append(MPNN_1(self.feat_dim, self.num_rels))
        if model_type == "mp2":
            self.layers.append(MPNN_2(feat_dim=self.feat_dim, num_rels=self.num_rels,
                                      num_wkr=self.num_wkr, num_tsk=self.num_tsk, e_dim=e_dim))
        elif model_type == "mp2_mp2":
            self.layers.append(MPNN_2(feat_dim=self.feat_dim, num_rels=self.num_rels,
                                      num_wkr=self.num_wkr, num_tsk=self.num_tsk, e_dim=e_dim))
            self.layers.append(MPNN_2(feat_dim=self.feat_dim, num_rels=self.num_rels,
                                      num_wkr=self.num_wkr, num_tsk=self.num_tsk, e_dim=e_dim))
        elif model_type == "mp2_mid_mp2":
            self.layers.append(MPNN_2(feat_dim=self.feat_dim, num_rels=self.num_rels,
                                      num_wkr=self.num_wkr, num_tsk=self.num_tsk, e_dim=e_dim))
            self.layers.append(middle_layer(self.feat_dim, self.num_wkr, self.num_tsk, num_heads=num_heads))
            self.layers.append(MPNN_2(feat_dim=self.feat_dim, num_rels=self.num_rels,
                                      num_wkr=self.num_wkr, num_tsk=self.num_tsk, e_dim=e_dim))

    def forward(self, features):

        if self.model_type in ["mp1", "mp2"]:
            embeddings = self.layers[0].forward(self.g, features)

        elif self.model_type in ["mp1_mp1", "mp2_mp2"]:
            embeddings = features
            for layer in self.layers:
                embeddings = layer(self.g, embeddings)

        elif self.model_type in ["mp1_mid_mp1", "mp2_mid_mp2"]:
            embeddings = self.layers[0].forward(self.g, features)
            embeddings = self.layers[1].forward(embeddings)
            embeddings = self.layers[2].forward(self.g, embeddings)

        return embeddings
