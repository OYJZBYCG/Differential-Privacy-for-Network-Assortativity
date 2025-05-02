#!/usr/bin/env python3
#coding:utf-8
import csv
import sys
import numpy as np
from scipy.sparse import lil_matrix


# Data File (input)
data_file = input('[FacebookFile (in)]:')
# Edge File (output)
edge_file = input('[EdgeFile (out)]:')
# Degree File (output)
deg_file = input('[DegFile (out)]:')

#################################### Main #####################################
# Read max_node_ID from the data file --> max_node_id
max_node_id = 0
with open(data_file, 'r') as file:
    for line in file:
        node1, node2 = map(int, line.split())
        if max_node_id < node1:
            max_node_id = node1
        if max_node_id < node2:
            max_node_id = node2
node_num = max_node_id + 1

# Read edges and degrees from the data file --> edges_lil, deg
edges_lil = lil_matrix((node_num, node_num))
deg = np.zeros(node_num)
f = open(data_file, 'r')
for line in f:
    lst = line.rstrip("\n").split(" ")
    node1 = int(lst[0])
    node2 = int(lst[1])
    edges_lil[node1, node2] = 1
    deg[node1] += 1
    deg[node2] += 1
f.close()

# Extract nodes with deg >= 1 and create new node IDs --> node_dic ({node_id, new_node_id})
node_dic = {}
new_node_id = 0
for node_id in range(node_num):
    if deg[node_id] > 0:
        node_dic[node_id] = new_node_id
        new_node_id += 1
print("#nodes:", len(node_dic))          

a1, a2 = edges_lil.nonzero()
print("#edges:", len(a1))         

# Output edge information
print("Outputting edge information.")
f = open(edge_file, "w")
print("#nodes", file=f)
print(len(node_dic), file=f)
print("node1,node2", file=f)
writer = csv.writer(f, lineterminator="\n")
for i in range(len(a1)):
    # node_ids --> node_id1, node_id2
    node_id1 = a1[i]
    node_id2 = a2[i]
    # new_node_ids --> node1, node2
    node1 = node_dic[node_id1]
    node2 = node_dic[node_id2]
    lst = [node1, node2]
    writer.writerow(lst)
f.close()

# Output degree information
print("Outputting degree information.")
f = open(deg_file, "w")
print("node,deg", file=f)
writer = csv.writer(f, lineterminator="\n")
for node_id in range(node_num):
    if deg[node_id] == 0:
        continue
    # new_node_id --> node
    node = node_dic[node_id]
    # new_node_id and its deg --> lst
    lst = [node, int(deg[node_id])]
    writer.writerow(lst)
f.close()