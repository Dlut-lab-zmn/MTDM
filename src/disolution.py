import re
from timeit import repeat
from pip import main
import torch
import sys
import numpy as np
sys.path.append("..")
from rgcn import utils
import os
dataset = 'YAGO'
print(os.path.exists(dataset))
def construct_data(heads_nodes,rels_nodes,tails_nodes,times_nodes):
    graph_dict = {}

    for i in range(len(heads_nodes)):
        graph_dict[i] = str(int(heads_nodes[i])) + '\t' + str( int(rels_nodes[i])) + '\t' + str(  int(tails_nodes[i])) + '\t' + str( int(times_nodes[i])) + '\n'
    return graph_dict

data = utils.load_data(dataset)
train_list = utils.split_by_time(data.train)

valid_list = utils.split_by_time(data.valid)
test_list = utils.split_by_time(data.test)

num_nodes = data.num_nodes
num_rels = data.num_rels
sub_total = num_rels * num_nodes
rel_total = num_rels

def get_list(graph):
    sub = graph[:,0]
    rel = graph[:,1]
    ob = graph[:,2]
    list_l = []
    for i in range(len(graph)):
        list_l.append([sub[i], rel[i], ob[i]])
    return list_l
def get_sub_list(graph):
    sub = graph[:,0]
    rel = graph[:,1]
    ob = graph[:,2]
    list_l = []
    list_r = []
    for i in range(len(graph)):
        list_l.append([sub[i], rel[i]])
        list_r.append([rel[i], ob[i]])
    return list_l,list_r
indx  = 9
for train_index in range(len(train_list)):
    train_index = train_index + 1
    if  train_index >indx:#train_index<len(train_list) - 10 and 
        heads_nodes = []
        rels_nodes = []
        tails_nodes = []
        times_nodes = []
        print(train_index,len(train_list))

        if train_index == 0:
            continue
        if train_index<10:
            his_graphs = train_list[:train_index-indx]
            fut_his_list = []
        else:
            his_graphs = train_list[train_index - 10:train_index-indx]
            fut_his_list = train_list[train_index - indx:train_index]
        locate = len(fut_his_list)
        fut_his_graph = False
        for i in range(locate):
            if i == 0:
                fut_his_graph = fut_his_list[i]
            else:
                fut_his_graph = np.concatenate((fut_his_graph, fut_his_list[i]),0)
        main_graph = train_list[train_index]
        if locate>1:
            list_fut_his_graph = get_list(fut_his_graph)
        list_main_graph = get_list(main_graph)
        list_l,list_r = get_sub_list(main_graph)
        for his_graph in his_graphs:
            for triple in his_graph:
                sub = triple[0]
                rel = triple[1]
                ob = triple[2]
                if locate>1:
                    if [sub, rel, ob] not in list_main_graph and [sub, rel, ob] not in list_fut_his_graph:
                        if  [sub, rel] in list_l:#  or [rel, ob] in list_r
                            heads_nodes.append(sub)
                            rels_nodes.append(rel)
                            tails_nodes.append(ob)
                            times_nodes.append(train_index)
                else:
                    if [sub, rel, ob] not in list_main_graph:
                        if  [sub, rel] in list_l:#  or [rel, ob] in list_r
                            heads_nodes.append(sub)
                            rels_nodes.append(rel)
                            tails_nodes.append(ob)
                            times_nodes.append(train_index)

        graph_dict = construct_data(heads_nodes,rels_nodes,tails_nodes,times_nodes)

        with open('../data/' + str(dataset) + '/train_data.txt', 'a+') as fp:
            for i in range(len(graph_dict)):
                fp.write(graph_dict[i])





