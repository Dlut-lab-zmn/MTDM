import re
from timeit import repeat
from pip import main
import torch
import sys
import numpy as np
sys.path.append("..")
from rgcn import utils
import os
dataset = 'ICEWS14s'
print(os.path.exists(dataset))
def construct_data(heads_nodes,rels_nodes,tails_nodes,times_nodes):
    graph_dict = {}

    for i in range(len(heads_nodes)):
        graph_dict[i] = str(int(heads_nodes[i])) + '\t' + str( int(rels_nodes[i])) + '\t' + str(  int(tails_nodes[i])) + '\t' + str( int(times_nodes[i])) + '\n'
    return graph_dict




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


data = utils.load_data(dataset)
train_list = utils.split_by_time(data.train)
valid_list = utils.split_by_time(data.valid)
test_list = utils.split_by_time(data.test)

"""
sub_ob_dict = {}
sub_ob_time_dict = {}
heads_nodes = []
rels_nodes = []
tails_nodes = []
times_nodes = []
for time, train_graph in enumerate(train_list):
    if time< len(train_list) - 10:
        time = time + 1
        for triple in train_graph:
            sub = triple[0]
            rel = triple[1]
            ob = triple[2]
            if str(sub) + '_' + str(ob) in sub_ob_dict.keys():
                if len(sub_ob_dict[str(sub) + '_' + str(ob) ]) >30:
                    neg_rel = sub_ob_dict[str(sub) + '_' + str(ob) ].pop(0)
                    neg_time = sub_ob_time_dict[str(sub) + '_' + str(ob) ].pop(0)
                    if neg_rel not in sub_ob_dict[str(sub) + '_' + str(ob) ]:
                        if time - neg_time > 10:
                            heads_nodes.append(sub)
                            rels_nodes.append(neg_rel)
                            tails_nodes.append(ob)
                            times_nodes.append(time)
            else:
                sub_ob_dict[str(sub) + '_' + str(ob) ] = []
                sub_ob_time_dict[str(sub) + '_' + str(ob) ] = []
            sub_ob_dict[str(sub) + '_' + str(ob) ].append(rel)
            sub_ob_time_dict[str(sub) + '_' + str(ob) ].append(time)
graph_dict = construct_data(heads_nodes,rels_nodes,tails_nodes,times_nodes)

with open('../data/' + str(dataset) + '/train_data30_10.txt', 'a+') as fp:
    for i in range(len(graph_dict)):
        fp.write(graph_dict[i])



total_sum = 0
sub_ob_val = []
for time, val_graph in enumerate(test_list):
        total_sum += len(val_graph)
        for triple in val_graph:
            sub = triple[0]
            rel = triple[1]
            ob = triple[2]
            if str(sub) + '_' +str(ob)+'_' + str(rel) not in sub_ob_val:
                sub_ob_val.append(str(sub) + '_' +str(ob)+'_' + str(rel))
sub_ob_train= []
print(total_sum)

total_sum = 0
for time, train_graph in enumerate(train_list):
        total_sum += len(train_graph)
        for triple in train_graph:
            sub = triple[0]
            rel = triple[1]
            ob = triple[2]
            if str(sub) + '_' +str(ob)+'_' + str(rel) not in sub_ob_train:
                sub_ob_train.append(str(sub) + '_' +str(ob)+'_' + str(rel))

statistic_sum = 0
for i in sub_ob_train:
    if i in sub_ob_val:
        statistic_sum += 1
print(total_sum,statistic_sum)

"""


sub_ob_dict = {}
sub_ob_time_dict = {}
heads_nodes = []
rels_nodes = []
tails_nodes = []
times_nodes = []
inter_time = 10
inter_rel = 10
for time, train_graph in enumerate(train_list):
    if time< len(train_list) - 1:
        time = time + 1
        for triple in train_graph:
            sub = triple[0]
            rel = triple[1]
            ob = triple[2]
            if str(sub) + '_' + str(ob) in sub_ob_dict.keys():
                length = len(sub_ob_dict[str(sub) + '_' + str(ob) ])
                if length >inter_rel:
                    for k in range(length - inter_rel):
                            neg_rel = sub_ob_dict[str(sub) + '_' + str(ob) ][0]
                            neg_time = sub_ob_time_dict[str(sub) + '_' + str(ob) ][0]
                            if time - neg_time > inter_time and neg_rel != rel and neg_rel not in sub_ob_time_dict[str(sub) + '_' + str(ob) ][1:]:
                                    sub_ob_dict[str(sub) + '_' + str(ob)] .pop(0)
                                    sub_ob_time_dict[str(sub) + '_' + str(ob) ].pop(0)
                                    heads_nodes.append(sub)
                                    rels_nodes.append(neg_rel)
                                    tails_nodes.append(ob)
                                    times_nodes.append(time)
                            else:
                                    sub_ob_dict[str(sub) + '_' + str(ob)] .pop(0)
                                    sub_ob_time_dict[str(sub) + '_' + str(ob) ].pop(0)
            else:
                sub_ob_dict[str(sub) + '_' + str(ob) ] = []
                sub_ob_time_dict[str(sub) + '_' + str(ob) ] = []
            
            if rel in sub_ob_dict[str(sub) + '_' + str(ob) ]:
                pos = sub_ob_dict[str(sub) + '_' + str(ob) ].index(rel)
                sub_ob_dict[str(sub) + '_' + str(ob) ].remove(rel)
                value = sub_ob_time_dict[str(sub) + '_' + str(ob) ][pos]
                sub_ob_time_dict[str(sub) + '_' + str(ob) ].remove(value)
            sub_ob_dict[str(sub) + '_' + str(ob) ].append(rel)
            sub_ob_time_dict[str(sub) + '_' + str(ob) ].append(time)
graph_dict = construct_data(heads_nodes,rels_nodes,tails_nodes,times_nodes)

with open('../data/' + str(dataset) + '/train_data.txt', 'w+') as fp:
    for i in range(len(graph_dict)):
        fp.write(graph_dict[i])
"""

heads_nodes = []
rels_nodes = []
tails_nodes = []
times_nodes = []
for time, train_graph in enumerate(test_list):
        for triple in train_graph:
            sub = triple[0]
            rel = triple[1]
            ob = triple[2]
            heads_nodes.append(sub)
            rels_nodes.append(rel)
            tails_nodes.append(ob)
            times_nodes.append(time + 335)
graph_dict = construct_data(heads_nodes,rels_nodes,tails_nodes,times_nodes)
with open('../data/' + str(dataset) + '/test.txt', 'w+') as fp:
    for i in range(len(graph_dict)):
        fp.write(graph_dict[i])

"""