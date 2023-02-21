import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# from rgcn.layers import RGCNBlockLayer as RGCNLayer
from rgcn.layers import ResUnionRGCNLayer,UnionRGCNLayer, RGCNBlockLayer
from src.model import BaseRGCN
from src.decoder import ConvTransE, ConvTransR, ConvTransE2
from GRU_module import GRUCell
from transformer import transformer
class RGCNCell(BaseRGCN):
    def build_hidden_layer(self, idx):
        act = F.rrelu
        if idx:
            self.num_basis = 0
        print("activate function: {}".format(act))
        if self.skip_connect:
            sc = False if idx == 0 else True
        else:
            sc = False
        if self.encoder_name == "uvrgcn":
            return UnionRGCNLayer(self.h_dim, self.h_dim, self.num_rels, self.num_bases,
                             activation=act, dropout=self.dropout, self_loop=self.self_loop, skip_connect=sc, rel_emb=self.rel_emb)
        else:
            raise NotImplementedError


    def forward(self, g, init_ent_emb, init_rel_emb):
        if self.encoder_name == "uvrgcn":
            node_id = g.ndata['id'].squeeze()
            g.ndata['h'] = init_ent_emb[node_id]
            x, r = init_ent_emb, init_rel_emb
            for i, layer in enumerate(self.layers):
                layer(g, [], r[i])
            return g.ndata.pop('h')
        else:
            if self.features is not None:
                print("----------------Feature is not None, Attention ------------")
                g.ndata['id'] = self.features
            node_id = g.ndata['id'].squeeze()
            g.ndata['h'] = init_ent_emb[node_id]
            if self.skip_connect:
                prev_h = []
                for layer in self.layers:
                    prev_h = layer(g, prev_h)
            else:
                for layer in self.layers:
                    layer(g, [])
            return g.ndata.pop('h')

class ResRGCNCell(BaseRGCN):
    def build_hidden_layer(self, idx):
        act = F.rrelu
        if idx:
            self.num_basis = 0
        print("activate function: {}".format(act))
        if self.skip_connect:
            sc = False if idx == 0 else True
        else:
            sc = False
        if self.encoder_name == "uvrgcn":
            return ResUnionRGCNLayer(self.h_dim, self.h_dim, self.num_rels, self.num_bases,
                             activation=act, dropout=self.dropout, self_loop=self.self_loop, skip_connect=sc, rel_emb=self.rel_emb)
        else:
            raise NotImplementedError


    def forward(self, g, init_ent_emb, init_rel_emb):
        if self.encoder_name == "uvrgcn":
            node_id = g.ndata['id'].squeeze()
            g.ndata['h'] = init_ent_emb[node_id]
            x, r = init_ent_emb, init_rel_emb
            for i, layer in enumerate(self.layers):
                layer(g, [], r[i])
            return g.ndata.pop('h')
        else:
            if self.features is not None:
                print("----------------Feature is not None, Attention ------------")
                g.ndata['id'] = self.features
            node_id = g.ndata['id'].squeeze()
            g.ndata['h'] = init_ent_emb[node_id]
            if self.skip_connect:
                prev_h = []
                for layer in self.layers:
                    prev_h = layer(g, prev_h)
            else:
                for layer in self.layers:
                    layer(g, [])
            return g.ndata.pop('h')

class RecurrentRGCN(nn.Module):
    def __init__(self, decoder_name, encoder_name, num_ents, num_rels, num_static_rels, num_words, h_dim, opn, sequence_len, num_bases=-1, num_basis=-1,
                 num_hidden_layers=1, dropout=0, self_loop=False, skip_connect=False, layer_norm=False, input_dropout=0,
                 hidden_dropout=0, feat_dropout=0, aggregation='cat', weight=1, discount=0, angle=0, use_static=False,use_global=False,
                 entity_prediction=False, relation_prediction=False, use_cuda=False,
                 gpu = 0, analysis=False):
        super(RecurrentRGCN, self).__init__()

        self.decoder_name = decoder_name
        self.encoder_name = encoder_name
        self.num_rels = num_rels
        self.num_ents = num_ents
        self.opn = opn
        self.num_words = num_words
        self.num_static_rels = num_static_rels
        self.sequence_len = sequence_len
        self.h_dim = h_dim
        self.layer_norm = layer_norm
        self.h = None
        self.run_analysis = analysis
        self.aggregation = aggregation
        self.relation_evolve = False
        self.weight = weight
        self.discount = discount
        self.use_static = use_static
        self.angle = angle
        self.relation_prediction = relation_prediction
        self.entity_prediction = entity_prediction
        self.emb_rel = None
        self.gpu = gpu
        self.use_global = use_global
        self.w1 = torch.nn.Parameter(torch.Tensor(self.h_dim, self.h_dim), requires_grad=True).float()
        torch.nn.init.xavier_normal_(self.w1)

        self.w2 = torch.nn.Parameter(torch.Tensor(self.h_dim, self.h_dim), requires_grad=True).float()
        torch.nn.init.xavier_normal_(self.w2)

        self.emb_rel = torch.nn.Parameter(torch.Tensor(self.num_rels * 2, self.h_dim), requires_grad=True).float()
        torch.nn.init.xavier_normal_(self.emb_rel)

        self.dynamic_emb = torch.nn.Parameter(torch.Tensor(num_ents, h_dim), requires_grad=True).float()
        torch.nn.init.normal_(self.dynamic_emb)

        if self.use_static:
            self.words_emb = torch.nn.Parameter(torch.Tensor(self.num_words, h_dim), requires_grad=True).float()
            torch.nn.init.xavier_normal_(self.words_emb)
            self.statci_rgcn_layer = RGCNBlockLayer(self.h_dim, self.h_dim, self.num_static_rels*2, num_bases,
                                                    activation=F.rrelu, dropout=dropout, self_loop=False, skip_connect=False)
            self.static_loss = torch.nn.MSELoss()
        if self.use_global:
            self.global_rgcn_layer = RGCNBlockLayer(self.h_dim, self.h_dim, self.num_rels*2, num_bases,
                                                    activation=F.rrelu, dropout=dropout, self_loop=True, skip_connect=False)

            self.static_loss = torch.nn.MSELoss()

        self.loss_r = torch.nn.CrossEntropyLoss() #self.loss_function#
        self.loss_e = torch.nn.CrossEntropyLoss()# self.loss_function#
        self.loss_e2 = self.loss_function# self.loss_function#


        self.rgcn = ResRGCNCell(num_ents,
                             h_dim,
                             h_dim,
                             num_rels * 2,
                             num_bases,
                             num_basis,
                             num_hidden_layers,
                             dropout,
                             self_loop,
                             skip_connect,
                             encoder_name,
                             self.opn,
                             self.emb_rel,
                             use_cuda,
                             analysis)

        self.rgcn2 = ResRGCNCell(num_ents,
                             h_dim,
                             h_dim,
                             num_rels * 2,
                             num_bases,
                             num_basis,
                             1,
                             dropout,
                             self_loop,
                             skip_connect,
                             encoder_name,
                             self.opn,
                             self.emb_rel,
                             use_cuda,
                             analysis)


        
        self.time_gate_weight = nn.Parameter(torch.Tensor(h_dim, h_dim))    
        nn.init.xavier_uniform_(self.time_gate_weight, gain=nn.init.calculate_gain('relu'))
        self.time_gate_bias = nn.Parameter(torch.Tensor(h_dim))
        nn.init.zeros_(self.time_gate_bias)     
                      
        self.gruh_gate_weight = nn.Parameter(torch.Tensor(h_dim, h_dim))
        nn.init.xavier_uniform_(self.gruh_gate_weight, gain=nn.init.calculate_gain('relu'))
        self.gruh_gate_bias = nn.Parameter(torch.Tensor(h_dim))
        nn.init.zeros_(self.gruh_gate_bias)

        self.tren_gate_weight = nn.Parameter(torch.Tensor(h_dim, h_dim))
        nn.init.xavier_uniform_(self.tren_gate_weight, gain=nn.init.calculate_gain('relu'))
        self.tren_gate_bias = nn.Parameter(torch.Tensor(h_dim))
        nn.init.zeros_(self.tren_gate_bias)


        # GRU cell for relation evolving
        # self.relation_cell_1 = nn.GRUCell(self.h_dim*2, self.h_dim)
        # self.h_cell_1 = nn.GRUCell(self.h_dim, self.h_dim)
        self.relation_cell_1 = GRUCell(self.h_dim*2, self.h_dim)
        self.h_cell_1 = GRUCell(self.h_dim, self.h_dim)
        # decoder
        if decoder_name == "convtranse":
            self.decoder_ob = ConvTransE(num_ents, h_dim, input_dropout, hidden_dropout, feat_dropout)
            self.rdecoder = ConvTransR(num_rels, h_dim, input_dropout, hidden_dropout, feat_dropout)
        else:
            raise NotImplementedError 

        self.glocal_transformer = transformer(dec_layer_num= 3 , embed_dim=h_dim, nhead= 4,
                                              dim_feedforward=224, dropout=0.1)


    def loss_function(self,x_input,y_target):
        softmax_func=nn.Softmax(dim=1)
        soft_output=softmax_func(x_input)
        log_output=torch.log(soft_output + 0.00001)

        index = torch.arange(len(y_target))
        #nlloss_output = - torch.mean(log_output[index,y_target])
        nlloss_output = - torch.mean(torch.unsqueeze(log_output[index,y_target],1) - 0.1*log_output[index])
        #nllloss_func=nn.NLLLoss()
        #nlloss_output=nllloss_func(log_output,y_target)
        return nlloss_output

    def loss_function2(self,x_input,y_target):
        softmax_func=nn.Softmax(dim=1)
        soft_output=softmax_func(x_input)
        log_output=torch.max(torch.log(soft_output + 0.00001), -torch.tensor(3.).cuda())

        index = torch.arange(len(y_target))
        nlloss_output = - torch.mean(log_output[index,y_target])
        #nlloss_output = - torch.mean(torch.unsqueeze(log_output[index,y_target],1) - 0.1*log_output[index])
        #nllloss_func=nn.NLLLoss()
        #nlloss_output=nllloss_func(log_output,y_target)
        return nlloss_output

    def forward(self, g_list, static_graph, global_graph,use_cuda,Formal):

        train_flag, memory_flag = Formal
        if train_flag:
            if memory_flag:
                self.use_static = False
                self.use_global = False
        if self.use_static:
            static_graph = static_graph.to(self.gpu)
            static_graph.ndata['h'] = torch.cat((self.dynamic_emb, self.words_emb), dim=0)
            self.statci_rgcn_layer(static_graph, [])
            static_emb = static_graph.ndata.pop('h')[:self.num_ents, :]
            static_emb = F.normalize(static_emb) if self.layer_norm else static_emb
            self.h = static_emb
        elif self.use_global:
            global_graph = global_graph.to(self.gpu)
            global_graph.ndata['h'] = self.dynamic_emb
            self.global_rgcn_layer(global_graph, [])
            global_emb = global_graph.ndata.pop('h')
            global_emb = F.normalize(global_emb) if self.layer_norm else global_emb
            self.h = global_emb
            if not self.use_static:
                static_emb = global_emb
        else:
            self.h = F.normalize(self.dynamic_emb) if self.layer_norm else self.dynamic_emb[:, :]
            static_emb = None
        history_embs = []
        calculate_embs = []
        self.h_0 = F.normalize(self.emb_rel) if self.layer_norm else self.emb_rel
        self.init_r = F.normalize(self.emb_rel) if self.layer_norm else self.emb_rel
        self.init_h = F.normalize(self.dynamic_emb) if self.layer_norm else self.dynamic_emb[:, :]
        self.gruh = self.h
        for i, g in enumerate(g_list):
            his_h = None

            g = g.to(self.gpu)

            current_h = self.rgcn.forward(g, self.gruh, [self.init_r, self.init_r])
            current_h = F.normalize(current_h) if self.layer_norm else current_h

            self.h = F.normalize(self.h_cell_1(current_h,self.h)) if self.layer_norm else self.h_cell_1(current_h,self.h)
            #self.h = F.normalize(self.glocal_transformer(current_h,self.h)) if self.layer_norm else self.glocal_transformer(current_h,self.h)

            gruh_weight = F.sigmoid(torch.mm(self.h, self.gruh_gate_weight) + self.gruh_gate_bias)
            self.h = gruh_weight * self.h + (1 - gruh_weight) * self.gruh

            if i == len(g_list) - 1:

                his_h = self.rgcn2.forward(g, self.init_h, [self.init_r, self.init_r])
                his_h = F.normalize(his_h) if self.layer_norm else his_h

                # tren_weight = F.sigmoid(torch.mm(his_h, self.tren_gate_weight) + self.tren_gate_bias)
                # his_h = tren_weight * his_h + (1 - tren_weight) * self.init_h

                if not train_flag:
                    time_weight = F.sigmoid(torch.mm(his_h, self.time_gate_weight) + self.time_gate_bias)
                    his_h = time_weight * his_h  + (1-time_weight) * self.h
                else:
                    if memory_flag:
                        time_weight = torch.ones(self.time_gate_bias.shape).to(self.gpu) if use_cuda else torch.zeros(self.time_gate_bias.shape)
                    else:
                        time_weight = torch.zeros(self.time_gate_bias.shape).to(self.gpu) if use_cuda else torch.ones(self.time_gate_bias.shape)
                    his_h = time_weight * his_h + (1 - time_weight) * self.h
                #print(torch.mean(time_weight))

            if his_h is not None:
                if memory_flag:
                    his_h = torch.cat((his_h,self.h),1)

                else:
                    his_h = torch.cat((his_h,his_h),1)

            else:
                his_h = torch.cat((self.h,self.h),1)
            history_embs.append(his_h)
            calculate_embs.append(self.h)

        return history_embs,calculate_embs, static_emb, self.h_0, torch.mean(time_weight)


    def predict(self, test_graph, num_rels, static_graph,global_graph, test_triplets,use_cuda,Formal):
        with torch.no_grad():
            inverse_test_triplets = test_triplets[:, [2, 1, 0]]
            inverse_test_triplets[:, 1] = inverse_test_triplets[:, 1] + num_rels 
            all_triples = torch.cat((test_triplets, inverse_test_triplets))

            evolve_embs, calculate_embs,static_emb, r_emb, _ = self.forward(test_graph, static_graph,global_graph,use_cuda,Formal)
            embedding = F.normalize(evolve_embs[-1]) if self.layer_norm else evolve_embs[-1]

            score = self.decoder_ob.forward(embedding, r_emb, all_triples, mode="test")
            score_rel = self.rdecoder.forward(embedding, r_emb, all_triples, mode="test")
            return all_triples, score, score_rel


    def get_loss(self, glist, triples,adv_triples, static_graph,global_graph,use_cuda, Formal):
        """
        :param glist:
        :param triplets:
        :param static_graph: 
        :param use_cuda:
        :return:
        """
        loss_ent = torch.zeros(1).cuda().to(self.gpu) if use_cuda else torch.zeros(1)
        loss_rel = torch.zeros(1).cuda().to(self.gpu) if use_cuda else torch.zeros(1)
        loss_static = torch.zeros(1).cuda().to(self.gpu) if use_cuda else torch.zeros(1)

        inverse_triples = triples[:, [2, 1, 0]]
        inverse_triples[:, 1] = inverse_triples[:, 1] + self.num_rels
        all_triples = torch.cat([triples, inverse_triples])
        all_triples = all_triples.to(self.gpu)

        if adv_triples is not None:
            # for (r, o) 2 s
            #adv_inverse_triples = adv_triples[:, [2, 1, 0]]
            #adv_inverse_triples[:, 1] = adv_inverse_triples[:, 1] + self.num_rels
            #all_adv_triples = torch.cat([adv_triples, adv_inverse_triples])
            #all_adv_triples = adv_inverse_triples.to(self.gpu)
            
            # for (s, r) 2 o
            all_adv_triples = adv_triples.to(self.gpu)
            



        evolve_embs,calculate_embs, static_emb, r_emb, time_weight = self.forward(glist, static_graph,global_graph,use_cuda,Formal)
        pre_emb = F.normalize(evolve_embs[-1]) if self.layer_norm else evolve_embs[-1]


        if self.entity_prediction:
            scores_ob = self.decoder_ob.forward(pre_emb, r_emb, all_triples).view(-1, self.num_ents)
            loss_ent += self.loss_e(scores_ob, all_triples[:, 2])
            if adv_triples is not None:
                scores_adv_ob = self.decoder_ob.forward(pre_emb, r_emb, all_adv_triples).view(-1, self.num_ents)
                loss_ent -= 0.01 * self.loss_e2(scores_adv_ob, all_adv_triples[:, 2])
        if self.relation_prediction:
            score_rel = self.rdecoder.forward(pre_emb, r_emb, all_triples, mode="train").view(-1, 2 * self.num_rels)
            loss_rel += self.loss_r(score_rel, all_triples[:, 1])
        loss_parm = time_weight * (1-time_weight)

        # # # regl
        # enti_emb = evolve_embs[-1]
        #
        # enti_emb = enti_emb[:, :enti_emb.shape[1]//2]
        # loss_reg = torch.norm(enti_emb,0)/(enti_emb.reshape((-1,1)).shape[0]) + torch.norm(r_emb,0)/(r_emb.reshape((-1,1)).shape[0])

        if self.use_static or self.use_global:
            if self.discount == 1:
                for time_step, evolve_emb in enumerate(calculate_embs):
                    step = (self.angle * math.pi / 180) * (time_step + 1)
                    if self.layer_norm:
                        sim_matrix = torch.sum(static_emb * F.normalize(evolve_emb[:,:static_emb.shape[1]]), dim=1)
                    else:
                        sim_matrix = torch.sum(static_emb * evolve_emb[:,:static_emb.shape[1]], dim=1)
                        c = torch.norm(static_emb, p=2, dim=1) * torch.norm(evolve_emb[:,:static_emb.shape[1]], p=2, dim=1)
                        sim_matrix = sim_matrix / c
                    mask = (math.cos(step) - sim_matrix) > 0
                    loss_static += self.weight * torch.sum(torch.masked_select(math.cos(step) - sim_matrix, mask))
            elif self.discount == 0:
                for time_step, evolve_emb in enumerate(calculate_embs):
                    step = (self.angle * math.pi / 180)
                    if self.layer_norm:
                        sim_matrix = torch.sum(static_emb * F.normalize(evolve_emb), dim=1)
                    else:
                        sim_matrix = torch.sum(static_emb * evolve_emb, dim=1)
                        c = torch.norm(static_emb, p=2, dim=1) * torch.norm(evolve_emb, p=2, dim=1)
                        sim_matrix = sim_matrix / c
                    mask = (math.cos(step) - sim_matrix) > 0
                    loss_static += self.weight * torch.sum(torch.masked_select(math.cos(step) - sim_matrix, mask))
        return loss_ent, loss_rel, loss_static,loss_parm,time_weight
