#!/usr/bin/env python3
import csv
import sys
import numpy as np
import networkx as nx

################################# Parameters ##################################
# Paramter n (#nodes)
N = int(input('[n (#nodes)]:'))
# Parameter m (#edges from a new node)
M = int(input('[m (#edges from a new node)]:'))
# Edge File (output)
EdgeFile = input('[EdgeFile (out)]:')
# Degree File (output)
DegFile = input('[DegFile (out)]:')

#################################### Main #####################################
# Fix a seed so that it depends on N
seed = N

# Generate a random graph according to the Barabasi-Albert model
G = nx.barabasi_albert_graph(N, M, seed)

# Calculate a degree for each node --> deg
deg = np.zeros(N, dtype=int)
for (i,j) in G.edges():
    deg[i] += 1
    deg[j] += 1

# Output edge information
print("Outputting edge information.")
f = open(EdgeFile, "w")
print("#nodes", file=f)
print(G.number_of_nodes(), file=f)
print("node1,node2", file=f)
writer = csv.writer(f, lineterminator="\n")
for (i,j) in G.edges():
    lst = [i, j]
    writer.writerow(lst)
f.close()

# Output degree information
print("Outputting degree information.")
f = open(DegFile, "w")
print("node,deg", file=f)
writer = csv.writer(f, lineterminator="\n")
for i in range(N):
    # actor index and her degree --> lst
    lst = [i, deg[i]]
    writer.writerow(lst)
f.close()
