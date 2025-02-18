#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import dgl
import dgl.nn as dglnn
import torch
from torch import nn
from torch.nn import functional as F

from cross_lingual import DOC_NODE_TYPE, WORD_NODE_TYPE


class DualAttentionHeteroGraphConv(nn.Module):
    """
    对每种边类型先做卷积，然后在类型级和节点级分别计算注意力权重，
    最后加权求和获得最终的节点表示。
    """
    def __init__(self, in_feats, out_feats, etypes):
        super().__init__()
        self.etypes = etypes
        self.conv_layers = nn.ModuleDict({
            t: dglnn.GraphConv(in_feats, out_feats) for t in etypes
        })

        self.a_type = nn.Parameter(torch.FloatTensor(2 * out_feats))
        nn.init.xavier_uniform_(self.a_type.unsqueeze(0))
        self.a_node = nn.Parameter(torch.FloatTensor(2 * out_feats))
        nn.init.xavier_uniform_(self.a_node.unsqueeze(0))
        self.leaky_relu = nn.LeakyReLU(0.2)

    def forward(self, graph, inputs):
        conv_out = {}
        for t in self.etypes:
            conv_out[t] = self.conv_layers[t](graph, inputs)
        h_stack = torch.stack([conv_out[t] for t in self.etypes], dim=0)

        mean_h = torch.mean(h_stack, dim=0, keepdim=True)
        scores = []
        for i in range(h_stack.size(0)):
            h_i = h_stack[i]
            score = self.leaky_relu((torch.cat([h_i, mean_h.squeeze(0)], dim=1) * self.a_type).sum(dim=1, keepdim=True))
            scores.append(score)
        scores = torch.stack(scores, dim=0)
        type_attn = torch.softmax(scores, dim=0)

        h_final = torch.sum(type_attn * h_stack, dim=0)
        return h_final



class HeteroTextGCN(nn.Module):
    def __init__(self, args, TYPE_LIST, dataset):
        super().__init__()
        self.valid_batch_size = args.valid_batch_size
        self.hidden_size = args.hidden_size
        self.out_emb_size = args.out_emb_size
        self.num_classes = args.num_classes
        self.device = args.device
        self.num_workers = args.num_workers
        n_layers = args.num_layers

        #self.conv1 = dglnn.HeteroGraphConv({
        #    t: dglnn.GraphConv(args.emb_size, args.hidden_size) for t in TYPE_LIST
        #}, aggregate="sum")
        #self.conv2 = dglnn.HeteroGraphConv({
        #    t: dglnn.GraphConv(args.hidden_size, args.out_emb_size) for t in TYPE_LIST
        #}, aggregate="sum")
        self.layers = nn.ModuleList()
        # 第一层：从原始 emb_size 到 hidden_size
        self.layers.append(DualAttentionHeteroGraphConv(args.emb_size, args.hidden_size, TYPE_LIST))
        for i in range(1, n_layers - 1):
            self.layers.append(DualAttentionHeteroGraphConv(args.hidden_size, args.hidden_size, TYPE_LIST))
            # 最后一层：hidden_size 到 out_emb_size
        self.layers.append(DualAttentionHeteroGraphConv(args.hidden_size, args.out_emb_size, TYPE_LIST))
        self.fc = nn.Linear(args.out_emb_size, args.num_classes)
        self.dropout = nn.Dropout(args.dropout)
        # self.layers = nn.ModuleList()
        # self.layers.append(dglnn.HeteroGraphConv({
        #     t: dglnn.GraphConv(args.emb_size, args.hidden_size) for t in TYPE_LIST
        #     }, aggregate="sum"))
        # for i in range(1, n_layers - 1):
        #     self.layers.append(dglnn.HeteroGraphConv({
        #         t: dglnn.GraphConv(args.hidden_size, args.hidden_size) for t in TYPE_LIST
        #         }, aggregate="sum"))
        # self.layers.append(dglnn.HeteroGraphConv({
        #     t: dglnn.GraphConv(args.hidden_size, args.out_emb_size) for t in TYPE_LIST
        #     }, aggregate="sum"))
        # self.fc = nn.Linear(args.out_emb_size, args.num_classes)
        # self.dropout = nn.Dropout(args.dropout)

    # def forward(self, g, blocks, inputs):
    #     h = inputs
    #     for l, (layer, block) in enumerate(zip(self.layers, blocks)):
    #         h = layer(block, h)
    #         if l != len(self.layers) - 1:
    #             h = {t: self.dropout(F.leaky_relu(h[t])) for t in h}
    #     h = {t: self.dropout(h[t]) for t in h}
    #     logits = self.fc(h[DOC_NODE_TYPE])
    #     #logits = h[DOC_NODE_TYPE]
    #     return h, logits

    def forward(self, g, blocks, inputs):
        h = inputs[DOC_NODE_TYPE]  # 假设只对文档节点做分类
        for l, layer in enumerate(self.layers):
            h = layer(g, {DOC_NODE_TYPE: h})
            if l != len(self.layers) - 1:
                h = self.dropout(torch.leaky_relu(h))
        logits = self.fc(h)
        return {DOC_NODE_TYPE: h}, logits

    def inference(self, g):
        nids = {}
        for ntype in g.ntypes:
            nids[ntype] = torch.arange(g.number_of_nodes(ntype))
        x = g.ndata["feat"]
        if not isinstance(x, dict):
            x = {DOC_NODE_TYPE: x}
        for l, layer in enumerate(self.layers):
            y = {t: torch.zeros(g.number_of_nodes(t), self.hidden_size if l != len(self.layers) - 1 else self.out_emb_size)
                    for t in x}

            sampler = dgl.dataloading.MultiLayerFullNeighborSampler(1)
            dataloader = dgl.dataloading.NodeDataLoader(
                g, nids, sampler,
                batch_size=self.valid_batch_size,
                shuffle=True,
                drop_last=False,
                num_workers=self.num_workers)

            for input_nodes, output_nodes, blocks in dataloader:
                if not isinstance(input_nodes, dict):
                    input_nodes = {DOC_NODE_TYPE: input_nodes.type(torch.long)}
                    output_nodes = {DOC_NODE_TYPE: output_nodes.type(torch.long)}
                block = blocks[0]
                block = block.int().to(self.device)
                h = {t: x[t][input_nodes[t]].to(self.device) for t in x}
                h = layer(block, h)
                if l != len(self.layers) - 1:
                    h = {t: self.dropout(F.leaky_relu(h[t])) for t in h}
                for t in h:
                    y[t][output_nodes[t]] = h[t].cpu()
            x = y

        return y[DOC_NODE_TYPE], self.fc(y[DOC_NODE_TYPE].to(self.device)).cpu()
