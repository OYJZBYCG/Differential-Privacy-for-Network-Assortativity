#!/usr/bin/env python3
#coding:utf-8
import csv
import sys
import numpy as np
from scipy.sparse import lil_matrix




# GitHub File (input)
DataFile = input('[GitHubFile (in)]:')
# Edge File (output)
EdgeFile = input('[EdgeFile (out)]:')
# Degree File (output)
DegFile = input('[DegFile (out)]:')


#################################### Main #####################################
# Read max_user_ID from the Orkut file --> max_user_id
f = open(DataFile, 'r')
max_user_id = 0
a = f.readline()
# print(a)
for i, line in enumerate(f):
    lst = line.rstrip("\n").split(",")
    user1 = int(lst[0])
    user2 = int(lst[1])
    if max_user_id < user1:
        max_user_id = user1
    if max_user_id < user2:
        max_user_id = user2
f.close()
user_num = max_user_id + 1
edges_lil = lil_matrix((user_num, user_num))
deg = np.zeros(user_num)

# Read edges and degrees from the DeezerEurope file --> edges_lil, deg
f = open(DataFile, 'r')
a = f.readline()
# print(a)
for i, line in enumerate(f):
    lst = line.rstrip("\n").split(",")
    user1 = int(lst[0])
    user2 = int(lst[1])
    edges_lil[user1, user2] = 1
    edges_lil[user2, user1] = 1
    deg[user1] += 1
    deg[user2] += 1
f.close()

# Extract users with deg >= 1 and create new user IDs --> user_dic ({user_id, new_user_id})
user_dic = {}
new_user_id = 0
for user_id in range(user_num):
    if deg[user_id] > 0:
        user_dic[user_id] = new_user_id
        new_user_id += 1
print("#users:", len(user_dic))        

#edges_csr = edges_lil.tocsr()
a1, a2 = edges_lil.nonzero()
#a1, a2 = edges_csr.nonzero()
print("#edges:", len(a1))        

# Output edge information
print("Outputting edge information.")
f = open(EdgeFile, "w")
print("#nodes", file=f)
print(len(user_dic), file=f)
print("node1,node2", file=f)
writer = csv.writer(f, lineterminator="\n")
for i in range(len(a1)):
    # user_ids --> user_id1, user_id2
    user_id1 = a1[i]
    user_id2 = a2[i]
    # new_user_ids --> user1, user2
    user1 = user_dic[user_id1]
    user2 = user_dic[user_id2]
    lst = [user1, user2]
    writer.writerow(lst)
f.close()

# Output degree information
print("Outputting degree information.")
f = open(DegFile, "w")
print("node,deg", file=f)
writer = csv.writer(f, lineterminator="\n")
for user_id in range(user_num):
    if deg[user_id] == 0:
        continue
    # new_user_id --> user1
    user1 = user_dic[user_id]
    # new_user_id and her deg --> lst
    lst = [user1, int(deg[user_id])]
    writer.writerow(lst)
f.close()