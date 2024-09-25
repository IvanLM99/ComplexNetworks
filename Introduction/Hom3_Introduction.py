# -*- coding: utf-8 -*-
"""
Created on Sun Oct 22 16:00:17 2023

@author: Ivan
"""

import csv
from operator import itemgetter
import networkx as nx
from networkx.algorithms import community

space = "-----------------------------------------------------------"

# Read in the nodelist file
with open('quakers_nodelist.csv', 'r') as nodecsv:
    nodereader = csv.reader(nodecsv)
    nodes = [n for n in nodereader][1:]
    
# Get a list of just the node names (the first item in each row)
node_names = [n[0] for n in nodes]

# Read in the edgelist file
with open('quakers_edgelist.csv', 'r') as edgecsv:
    edgereader = csv.reader(edgecsv)
    edges = [tuple(e) for e in edgereader][1:]

G = nx.Graph() # Initialize a Graph object
G.add_nodes_from(node_names) # Add nodes to the Graph
G.add_edges_from(edges) # Add edges to the Graph
print(G) # Print information about the Graph
print(space)

# Create an empty dictionary for each attribute
hist_sig_dict = {}
gender_dict = {}
birth_dict = {}
death_dict = {}
id_dict = {}

for node in nodes: # Loop through the list of nodes, one row at a time
    hist_sig_dict[node[0]] = node[1] # Access the correct item, add it to the corresponding dictionary
    gender_dict[node[0]] = node[2]
    birth_dict[node[0]] = node[3]
    death_dict[node[0]] = node[4]
    id_dict[node[0]] = node[5]

# Add each dictionary as a node attribute to the Graph object
nx.set_node_attributes(G, hist_sig_dict, 'historical_significance')
nx.set_node_attributes(G, gender_dict, 'gender')
nx.set_node_attributes(G, birth_dict, 'birth_year')
nx.set_node_attributes(G, death_dict, 'death_year')
nx.set_node_attributes(G, id_dict, 'sdfb_id')

#Density -> ratio of edges to all nodes
density = nx.density(G)
print("Network density:", density)
print(space)

# Diamater -> longest of all shortest past
# If your Graph has more than one component, this will return False:
print(nx.is_connected(G))
print(space)

# Next, use nx.connected_components to get the list of components,
# then use the max() command to find the largest one:
components = nx.connected_components(G)
largest_component = max(components, key=len)

# Create a "subgraph" of just the largest component
# Then calculate the diameter of the subgraph, just like you did with density.
subgraph = G.subgraph(largest_component)
diameter = nx.diameter(subgraph)
print("Network diameter of largest component:", diameter)
print(space)

# Transitivity -> ratio of triangles
triadic_closure = nx.transitivity(G)
print("Triadic closure:", triadic_closure)
print(space)

# Degree -> commonway of findingimportant nodes (sum of edges)
degree_dict = dict(G.degree(G.nodes()))
nx.set_node_attributes(G, degree_dict, 'degree')
#print(G.nodes['William Penn'])
# Sort -> ordenar diccionario
sorted_degree = sorted(degree_dict.items(), key=itemgetter(1), reverse=True)

print("Top 20 nodes by degree:",'\n')
for d in sorted_degree[:20]:
    print(d)
print(space)

# Betweenness centrality  
# Eigenvector centrality 
betweenness_dict = nx.betweenness_centrality(G) # Run betweenness centrality
eigenvector_dict = nx.eigenvector_centrality(G) # Run eigenvector centrality

# Assign each to an attribute in your network
nx.set_node_attributes(G, betweenness_dict, 'betweenness')
nx.set_node_attributes(G, eigenvector_dict, 'eigenvector')

sorted_betweenness = sorted(betweenness_dict.items(), key=itemgetter(1), reverse=True)
print("Top 20 nodes by betweenness centrality:",'\n')
for b in sorted_betweenness[:20]:
    print(b)
print(space)

#First get the top 20 nodes by betweenness as a list
top_betweenness = sorted_betweenness[:20]

#Then find and print their degree
for tb in top_betweenness: # Loop through top_betweenness
    degree = degree_dict[tb[0]] # Use degree_dict to access a node's degree
    print("Name:", tb[0], "| Betweenness Centrality:", tb[1], "| Degree:", degree)
print(space)

# Community -> subgroups of families in a network
# Modularity -> relative density in a community
communities = community.greedy_modularity_communities(G)
modularity_dict = {} # Create a blank dictionary
for i,c in enumerate(communities): # Loop through the list of communities, keeping track of the number for the community
    for name in c: # Loop through each person in a community
        modularity_dict[name] = i # Create an entry in the dictionary for the person, where the value is which group they belong to.

# Now you can add modularity information like we did the other metrics
nx.set_node_attributes(G, modularity_dict, 'modularity')

# First get a list of just the nodes in that class
class0 = [n for n in G.nodes() if G.nodes[n]['modularity'] == 0]

# Then create a dictionary of the eigenvector centralities of those nodes
class0_eigenvector = {n:G.nodes[n]['eigenvector'] for n in class0}

# Then sort that dictionary and print the first 5 results
class0_sorted_by_eigenvector = sorted(class0_eigenvector.items(), key=itemgetter(1), reverse=True)

print("Modularity Class 0 Sorted by Eigenvector Centrality:",'\n')
for node in class0_sorted_by_eigenvector[:5]:
    print("Name:", node[0], "| Eigenvector Centrality:", node[1])
print(space)
    
for i,c in enumerate(communities): # Loop through the list of communities
    if len(c) > 2: # Filter out modularity classes with 2 or fewer nodes
        print('Class '+str(i)+':', list(c)) # Print out the classes and their members
print(space)

# Que falta del capitulo 7 por hacer?

# Katz Centrality

# alpha = 0.85  
# katz_centrality = nx.katz_centrality(subgraph, alpha=alpha, max_iter=1000)
# # Assign to an attribute in your network
# nx.set_node_attributes(subgraph, katz_centrality, 'katz_centrality')
# sorted_katzcen = sorted(katz_centrality.items(), key=itemgetter(1), reverse=True)
# print("Top 20 nodes by katz centrality:")
# for b in sorted_katzcen[:20]:
#     print(b)
# print(space)


# PageRank centrality

pagerank_centrality = nx.pagerank(G, alpha=0.85)  # You can adjust the alpha value if needed
# Assign PageRank centrality as a node attribute
nx.set_node_attributes(G, pagerank_centrality, 'pagerank_centrality')
# Sort the nodes by PageRank centrality
sorted_pagerank = sorted(pagerank_centrality.items(), key=itemgetter(1), reverse=True)

# Print the top 20 nodes by PageRank centrality
print("Top 20 nodes by PageRank centrality:",'\n')
for node, centrality in sorted_pagerank[:20]:
    print(f"Node {node}: PageRank Centrality = {centrality}")
print(space)

# Closeness Centrality

closeness_centrality = nx.closeness_centrality(subgraph)
# Assign Closeness Centrality as a node attribute
nx.set_node_attributes(subgraph, closeness_centrality, 'closeness_centrality')
# Sort the nodes by Closeness Centrality
sorted_closeness = sorted(closeness_centrality.items(), key=itemgetter(1), reverse=True)

# Print the top 20 nodes by Closeness Centrality
print("Top 20 nodes by Closeness Centrality:",'\n')
for node, centrality in sorted_closeness[:20]:
    print(f"Node {node}: Closeness Centrality = {centrality}")
print(space)


# Clique Number (Maximal Cliques)
cliques = list(nx.find_cliques(subgraph))
clique_number = max(len(clique) for clique in cliques)
print(f"Clique Number (Maximal Cliques): {clique_number}")
print(space)

# K-Core Number
k_core_number = nx.core_number(subgraph)
print("K-Core Numbers:",'\n')
for node, k_core in k_core_number.items():
    print(f"Node {node}: K-Core Number = {k_core}")
print(space)

# Clustering Coefficient
clustering_coefficient = nx.average_clustering(subgraph)
print(f"Average Clustering Coefficient: {clustering_coefficient}")
print(space)

# Redundancy
# Initialize a dictionary to store the redundancy values for each node
redundancy_dict = {}

# Calculate redundancy for each node
for node in subgraph.nodes():
    neighbors = list(subgraph.neighbors(node))
    total_redundancy = 0
    for neighbor in neighbors:
        neighbor_neighbors = list(subgraph.neighbors(neighbor))
        neighbor_neighbors.remove(node)  # Exclude the current node itself
        total_redundancy += len(set(neighbor_neighbors) & set(neighbors))
    redundancy = total_redundancy / len(neighbors) if len(neighbors) > 0 else 0
    redundancy_dict[node] = redundancy

# Print the redundancy values for each node
print("Redundancy Values:",'\n')
for node, redundancy in redundancy_dict.items():
    print(f"Node {node}: Redundancy = {redundancy}")
print(space)

# Reciprocity

reciprocity = nx.reciprocity(subgraph)
print(f"Reciprocity: {reciprocity}")
print(space)

# Similarity - structural equivalence (simplest version)
structural_equivalence = {}
for node in G.nodes():
    neighbors = set(G.neighbors(node))
    equivalent_nodes = [n for n in G.nodes() if set(G.neighbors(n)) == neighbors and n != node]
    structural_equivalence[node] = equivalent_nodes

# Print nodes with structural equivalences
print("Nodes with structural equivalences:", '\n')
for node, equivalent_nodes in structural_equivalence.items():
    if equivalent_nodes:
        print(f"Node {node} is structurally equivalent to nodes: {equivalent_nodes}")
print(space)
