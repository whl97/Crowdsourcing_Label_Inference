import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class MPNN_1(nn.Module):
    def __init__(self, feat_dim, num_rels):
        super(MPNN_1, self).__init__()
        self.feat_dim = feat_dim
        self.num_rels = num_rels
        self.weight_r = nn.Parameter(
            torch.Tensor(num_rels, feat_dim, feat_dim))
        nn.init.xavier_uniform_(
            self.weight_r, gain=nn.init.calculate_gain('relu'))

    def forward(self, g, features):
        def msg_func(edges):
            w_r = self.weight_r[edges.data['type'].long()]
            h_j = edges.src['h']
            msg = torch.div(torch.matmul(h_j, w_r), edges.dst['deg'])
            return {'msg': msg}

        def red_func(nodes):
            c = 0.2
            M_i = torch.sum(nodes.mailbox['msg'], dim=1)
            h_i = nodes.data['h']
            return {'h': F.relu(c * h_i + (1-c)*M_i)}

        g.ndata['h'] = features
        g.register_message_func(msg_func)
        g.register_reduce_func(red_func)
        g.update_all()
        return g.ndata.pop('h')


class MPNN_2(nn.Module):
    def __init__(self, feat_dim, num_rels, num_wkr, num_tsk, e_dim):
        super(MPNN_2, self).__init__()
        self.feat_dim = feat_dim
        self.e_dim = e_dim
        self.num_wkr = num_wkr
        self.num_tsk = num_tsk

        self.w1_wkr = nn.Parameter(
            torch.Tensor(feat_dim + self.e_dim, feat_dim))
        self.b1_wkr = nn.Parameter(torch.Tensor(1, feat_dim))
        nn.init.xavier_uniform_(
            self.w1_wkr, gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_uniform_(
            self.b1_wkr, gain=nn.init.calculate_gain('relu'))

        self.w1_tsk = nn.Parameter(
            torch.Tensor(feat_dim + self.e_dim, feat_dim))
        self.b1_tsk = nn.Parameter(torch.Tensor(1, feat_dim))
        nn.init.xavier_uniform_(
            self.w1_tsk, gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_uniform_(
            self.b1_tsk, gain=nn.init.calculate_gain('relu'))

        self.w2_wkr = nn.Parameter(torch.Tensor(feat_dim, feat_dim))
        self.b2_wkr = nn.Parameter(torch.Tensor(1, feat_dim))
        nn.init.xavier_uniform_(
            self.w2_wkr, gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_uniform_(
            self.b2_wkr, gain=nn.init.calculate_gain('relu'))

        self.w2_tsk = nn.Parameter(torch.Tensor(feat_dim, feat_dim))
        self.b2_tsk = nn.Parameter(torch.Tensor(1, feat_dim))
        nn.init.xavier_uniform_(
            self.w2_tsk, gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_uniform_(
            self.b2_tsk, gain=nn.init.calculate_gain('relu'))

        self.w3_wkr = nn.Parameter(torch.Tensor(feat_dim * 2, 1))
        self.b3_wkr = nn.Parameter(torch.Tensor(1, 1))
        nn.init.xavier_uniform_(
            self.w3_wkr, gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_uniform_(
            self.b3_wkr, gain=nn.init.calculate_gain('relu'))

        self.w3_tsk = nn.Parameter(torch.Tensor(feat_dim * 2, 1))
        self.b3_tsk = nn.Parameter(torch.Tensor(1, 1))
        nn.init.xavier_uniform_(
            self.w3_tsk, gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_uniform_(
            self.b3_tsk, gain=nn.init.calculate_gain('relu'))

        self.e_wkr = nn.Parameter(torch.Tensor(num_rels, 1, self.e_dim))
        nn.init.xavier_uniform_(
            self.e_wkr, gain=nn.init.calculate_gain('relu'))
        self.e_tsk = nn.Parameter(torch.Tensor(num_rels, 1, self.e_dim))
        nn.init.xavier_uniform_(
            self.e_tsk, gain=nn.init.calculate_gain('relu'))

    def forward(self, g, features):
        g.ndata['h'] = features

        def msg_func_wkr(edges):
            h_j = edges.src['h'].squeeze(1)
            e_ij = self.e_wkr[edges.data['type'].long()].squeeze(1)
            msg = torch.sigmoid(
                (torch.matmul(torch.cat((h_j, e_ij), dim=1), self.w1_wkr)).unsqueeze(
                    1) + self.b1_wkr)
            return {'msg': msg}

        def red_func_wkr(nodes):
            M_j = nodes.mailbox['msg'].squeeze(2)
            h_i = nodes.data['h']

            alpha = F.softmax(
                (torch.matmul(torch.cat((M_j, h_i.repeat(1, M_j.shape[1], 1)), dim=2), self.w3_wkr).unsqueeze(
                    1) + self.b3_wkr).squeeze(1), dim=1)

            hi_new = torch.sigmoid(
                torch.matmul((torch.sum(M_j.mul(alpha), dim=1)) + 0.5 * h_i.squeeze(1), self.w2_wkr) + self.b2_wkr)

            return {'h': hi_new.unsqueeze(1)}

        def msg_func_tsk(edges):
            h_j = edges.src['h'].squeeze(1)
            e_ij = self.e_tsk[edges.data['type'].long()].squeeze(1)
            msg = torch.sigmoid(
                (torch.matmul(torch.cat((h_j, e_ij), dim=1), self.w1_tsk)).unsqueeze(
                    1) + self.b1_tsk)
            return {'msg': msg}

        def red_func_tsk(nodes):
            c = 0.3
            M_j = nodes.mailbox['msg'].squeeze(2)
            h_i = nodes.data['h']

            alpha = F.softmax(
                (torch.matmul(torch.cat((M_j, h_i.repeat(1, M_j.shape[1], 1)), dim=2), self.w3_tsk).unsqueeze(
                    1) + self.b3_tsk).squeeze(1), dim=1)

            hi_new = torch.sigmoid(
                torch.matmul(((1 - c) * torch.sum(M_j.mul(alpha), dim=1)) + c * h_i.squeeze(1),
                             self.w2_tsk) + self.b2_tsk)

            return {'h': hi_new.unsqueeze(1)}

        g.register_message_func(msg_func_wkr)
        g.register_reduce_func(red_func_wkr)
        g.pull(g.nodes()[range(self.num_wkr)])

        g.register_message_func(msg_func_tsk)
        g.register_reduce_func(red_func_tsk)
        g.pull(g.nodes()[range(self.num_wkr, self.num_wkr + self.num_tsk)])
        return g.ndata.pop('h')


class self_attention_single_head(nn.Module):
    def __init__(self, feat_dim, num_wkr, num_tsk):
        super(self_attention_single_head, self).__init__()
        self.feat_dim = feat_dim
        self.num_wkr = num_wkr
        self.num_tsk = num_tsk

        self.projection_q_wkr = nn.Linear(feat_dim, feat_dim)
        self.projection_k_wkr = nn.Linear(feat_dim, feat_dim)
        self.projection_v_wkr = nn.Linear(feat_dim, feat_dim)

        self.projection_q_tsk = nn.Linear(feat_dim, feat_dim)
        self.projection_k_tsk = nn.Linear(feat_dim, feat_dim)
        self.projection_v_tsk = nn.Linear(feat_dim, feat_dim)


    def forward(self, features):
        wkr_features = features[0:self.num_wkr]
        tsk_features = features[self.num_wkr: self.num_wkr + self.num_tsk]

        wkr_query = self.projection_q_wkr(wkr_features)
        wkr_key = self.projection_k_wkr(wkr_features)
        wkr_value = self.projection_v_wkr(wkr_features)
        new_wkr_features = self.dot_product_attention(
            wkr_query, wkr_key, wkr_value)

        tsk_query = self.projection_q_tsk(tsk_features)
        tsk_key = self.projection_k_tsk(tsk_features)
        tsk_value = self.projection_v_tsk(tsk_features)
        new_tsk_features = self.dot_product_attention(
            tsk_query, tsk_key, tsk_value)

        new_feature = torch.cat([new_wkr_features, new_tsk_features], dim=-2)

        return new_feature

    def dot_product_attention(self, query, key, value):
        d_k = query.size(-1)
        key_ = key.transpose(-2, -1)
        scores = torch.matmul(query, key_)/math.sqrt(d_k)
        weights_att = F.softmax(scores, dim=-1)
        return torch.matmul(weights_att, value)

class middle_layer(nn.Module):
    def __init__(self, feat_dim, num_wkr, num_tsk, num_heads=1):
        super(middle_layer, self).__init__()
        self.feat_dim = feat_dim
        self.num_wkr = num_wkr
        self.num_tsk = num_tsk
        self.num_nodes = num_wkr+num_tsk
        self.num_heads = num_heads
        self.heads = nn.ModuleList()
        for _ in range(num_heads):
            self.heads.append(self_attention_single_head(
                feat_dim, num_wkr, num_tsk)) 
        self.projection = nn.Linear(num_heads*self.feat_dim, feat_dim)

    def forward(self, features):
        features = features.squeeze(1)

        multiheads_output_list = []
        for head in self.heads:
            multiheads_output_list.append(head(features))
        outputs = torch.cat(multiheads_output_list, dim=-1)
        outputs = self.projection(outputs)

        return outputs.unsqueeze(1)
