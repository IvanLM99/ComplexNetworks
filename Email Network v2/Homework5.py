# database from https://snap.stanford.edu/data/email-Eu-core-temporal.html

import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from math import e, log
import collections
import random

# Load the dataset
file_path = 'email-Eu-core-temporal.txt'
graph = nx.Graph()

# Read the dataset and add edges to the graph
with open(file_path, 'r') as file:
    for line in file:
        src, tgt, _ = map(int, line.strip().split())  
        graph.add_edge(src, tgt)

# Get number of nodes and edges
num_nodes = graph.number_of_nodes()
num_edges = graph.number_of_edges()

# Calculate edge probability for Erdos-Renyi model
p = 2*num_edges / (num_nodes * (num_nodes - 1)) if num_nodes > 1 else 0

# Generate 5 random ER networks with the same number of nodes and edge probability
er_networks = []
for _ in range(5):
    er_graph = nx.erdos_renyi_graph(num_nodes, p)
    er_networks.append(er_graph)

# Get degree sequence from the original graph
degree_sequence = [d for n, d in graph.degree()]

# Generate 5 random CM networks with the same degree sequence
cm_networks = []
for _ in range(5):
    # Shuffle the degree sequence to create variations
    random.shuffle(degree_sequence)
    cm_graph = nx.configuration_model(degree_sequence)
    cm_networks.append(cm_graph)

# Plot Erdos-Renyi networks
plt.figure(figsize=(15, 5))
plt.suptitle('Erdos-Renyi Networks')

for i, er_graph in enumerate(er_networks, 1):
    plt.subplot(1, 5, i)
    nx.draw(er_graph, node_color='skyblue', node_size=50)
    plt.title(f'ER Network {i}')

plt.tight_layout()
plt.show()

# Plot Configuration Model networks
plt.figure(figsize=(15, 5))
plt.suptitle('Configuration Model Networks')

for i, cm_graph in enumerate(cm_networks, 1):
    plt.subplot(1, 5, i)
    nx.draw(cm_graph, node_color='lightgreen', node_size=50)
    plt.title(f'CM Network {i}')

plt.tight_layout()
plt.show()

# Get the degree sequence from the real network
degree_sequence_real = [d for n, d in graph.degree()]

# Generate a Configuration Model network with the same degree sequence
cm_graph = nx.configuration_model(degree_sequence_real)

# Calculate degree sequence of the CM network
degree_sequence_cm = [d for n, d in cm_graph.degree()]

# Plot histograms to compare degree distributions
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.hist(degree_sequence_real, bins='auto', color='skyblue', alpha=0.7)
plt.title('Degree Distribution of Real Network')
plt.xlabel('Degree')
plt.ylabel('Frequency')

plt.subplot(1, 2, 2)
plt.hist(degree_sequence_cm, bins='auto', color='lightgreen', alpha=0.7)
plt.title('Degree Distribution of CM Network')
plt.xlabel('Degree')
plt.ylabel('Frequency')

plt.tight_layout()
plt.show()

# Calculate average degree of nodes and their neighbors in the real network
avg_degree_real = sum(dict(graph.degree()).values()) / len(graph.nodes())

# Calculate average degree of neighbors
avg_neighbor_degree_real = sum(graph.degree(nbr) for n, nbr in graph.edges()) / len(graph.edges())

print("Real Network:")
print(f"Average Degree of Nodes: {avg_degree_real}")
print(f"Average Degree of Neighbors: {avg_neighbor_degree_real}\n")

# Generate Erdos-Renyi network with the same number of nodes and edge probability as the real network
num_nodes = graph.number_of_nodes()
num_edges = graph.number_of_edges()
p = num_edges / (num_nodes * (num_nodes - 1) / 2) if num_nodes > 1 else 0

er_graph = nx.erdos_renyi_graph(num_nodes, p)

# Calculate average degree of nodes and their neighbors in the Erdos-Renyi network
avg_degree_er = sum(dict(er_graph.degree()).values()) / len(er_graph.nodes())
avg_neighbor_degree_er = sum(er_graph.degree(nbr) for n, nbr in er_graph.edges()) / len(er_graph.edges())

print("Erdos-Renyi Network:")
print(f"Average Degree of Nodes: {avg_degree_er}")
print(f"Average Degree of Neighbors: {avg_neighbor_degree_er}\n")

# Generate Configuration Model network with the same degree sequence as the real network
degree_sequence = [d for n, d in graph.degree()]
cm_graph = nx.configuration_model(degree_sequence)

# Calculate average degree of nodes and their neighbors in the Configuration Model network
avg_degree_cm = sum(dict(cm_graph.degree()).values()) / len(cm_graph.nodes())
avg_neighbor_degree_cm = sum(cm_graph.degree(nbr) for n, nbr in cm_graph.edges()) / len(cm_graph.edges())

print("Configuration Model Network:")
print(f"Average Degree of Nodes: {avg_degree_cm}")
print(f"Average Degree of Neighbors: {avg_neighbor_degree_cm}")

