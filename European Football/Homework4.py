# database from https://snap.stanford.edu/data/email-Eu-core-temporal.html

import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from math import e, log
import collections

# Load the dataset
file_path = 'email-Eu-core-temporal.txt'
graph = nx.Graph()

# Read the dataset and add edges to the graph
with open(file_path, 'r') as file:
    for line in file:
        src, tgt, _ = map(int, line.strip().split())  
        graph.add_edge(src, tgt)

# Calculate the degree distribution
degree_sequence = [d for n, d in graph.degree()]

# Print basic network information
print(f"Number of nodes: {graph.number_of_nodes()}")
print(f"Number of edges: {graph.number_of_edges()}")
print(f"Density: {nx.density(graph)}")
print(f"Is connected: {nx.is_connected(graph)}")
print(f"Average degree: {2 * graph.number_of_edges() / graph.number_of_nodes()}")

# Create three separate plots
plt.figure(figsize=(6, 6))

# Plot 0 - Network Node Representation
plt.figure(figsize=(10, 8))
pos = nx.spring_layout(graph, seed=42)
nx.draw(graph, pos, with_labels=False, node_size=10)
plt.title("Network Node Representation")
plt.show()

# Plot 1: Histogram
plt.hist(degree_sequence, bins=50, alpha=0.7, color='b', edgecolor='k', density=True)
plt.title('Histogram')
plt.xlabel('Degree')
plt.ylabel('Fraction of nodes with degree k')
plt.grid(True)
plt.show()

plt.figure(figsize=(6, 6))

# Plot 2: Histogram in log-log scale
plt.hist(degree_sequence, bins=50, alpha=0.7, color='b', edgecolor='k', density=True)
plt.title('Histogram (Log-Log Scale)')
plt.xlabel('Degree')
plt.ylabel('Fraction of nodes with degree k')
plt.xscale('log')
plt.yscale('log')
plt.grid(True)
plt.show()

# Plot 3: Create a plot with the degree distribution in log-log scale
plt.figure(figsize=(8, 8))

# Calculate alpha and kmin
kmin = min(degree_sequence)
N = sum(1 for k in degree_sequence if k >= kmin)
alpha = 1 + N * sum(np.log(k / (kmin - 0.5))**-1 for k in degree_sequence if k >= kmin)

# Define a range of 'c' values
c_values = [0.1, 0.5, 1.0, 1.5, 2.0]

for c in c_values:
    C = e**c
    pk = [C * (k**(-alpha)) for k in degree_sequence]

    # Plot the degree distribution using the formula in log-log scale
    hist, bin_edges = np.histogram(degree_sequence, bins=50, density=True)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    plt.loglog(bin_centers, hist, label=f'c = {c}')

plt.title('Degree Distribution (Log-Log Scale) for Different c Values')
plt.xlabel('Degree (log scale)')
plt.ylabel('Fraction of nodes with degree k (log scale)')
plt.legend()
plt.grid(True)
plt.show()

# Plot 4: Create a single plot with the degree distribution in log-log scale
plt.figure(figsize=(8, 8))

# Define the number of bins and the range
num_bins = 50
min_degree = min(degree_sequence)
max_degree = max(degree_sequence)

# Choose the base 'a' for logarithmic binning (e.g., a=2)
a = 2

# Calculate bin edges based on logarithmic binning
bin_edges = [a ** n for n in range(int(np.log(min_degree) / np.log(a)), int(np.log(max_degree) / np.log(a)) + 1)]
widths = [bin_edges[i + 1] - bin_edges[i] for i in range(len(bin_edges) - 1)]

# Calculate the histogram with logarithmic binning
hist = np.histogram(degree_sequence, bins=bin_edges)

# Normalize by bin width
hist_norm = hist[0] / widths

# Plot the histogram with logarithmic binning and black borders
plt.bar(bin_edges[:-1], hist_norm, widths, align='edge', edgecolor='black')
plt.xscale('log')
plt.yscale('log')

plt.title('Histogram with Logarithmic Binning (Log-Log Scale)')
plt.xlabel('Degree (log scale)')
plt.ylabel('Normalized fraction of nodes with degree k')
plt.grid(True)
plt.show()

# Plot 5: cummulative distribution function for the degrees of nodes on the network
# Sort the degrees in ascending order
sorted_degrees = np.sort(degree_sequence)

# Calculate the CDF values
cdf = np.arange(1, len(sorted_degrees) + 1) / len(sorted_degrees)

# Plot the CDF on logarithmic scales
plt.figure(figsize=(8, 6))
plt.loglog(sorted_degrees, 1 - cdf, marker='o', linestyle='-', color='b')
plt.grid(True)

plt.title('Cumulative Distribution Function of Node Degrees (Log-Log Scale)')
plt.xlabel('Degree (log scale)')
plt.ylabel('Commulative distribution of fraction of nodes with degree k (log scale)')
plt.show()

print("============================================")

# Get the degrees of all nodes in the graph
node_degrees = list(graph.degree())

# Sort nodes by their degrees in descending order
sorted_nodes = sorted(node_degrees, key=lambda x: x[1], reverse=True)

# Print the nodes with the highest degrees (top 10)
top_nodes = sorted_nodes[:10]  # You can change the number to print more or fewer nodes

for node, degree in top_nodes:
    print(f"Node {node}: Degree {degree}")

print("============================================")

# Calculate Degree Centrality
degree_centrality = nx.degree_centrality(graph)

# Sort nodes by their degree centrality in descending order
sorted_nodes = sorted(degree_centrality.items(), key=lambda x: x[1], reverse=True)

# Print the nodes with the highest degree centrality (top 10 in this example)
top_nodes = sorted_nodes[:10]  # You can change the number to print more or fewer nodes

for node, degree in top_nodes:
    print(f"Node {node}: Degree Centrality: {degree:.4f}")
    
print("============================================")
    
# Calculate Closeness Centrality
closeness_centrality = nx.closeness_centrality(graph)

# Sort nodes by their closeness centrality in descending order
sorted_nodes = sorted(closeness_centrality.items(), key=lambda x: x[1], reverse=True)

# Print the nodes with the highest closeness centrality (top 10 in this example)
top_nodes = sorted_nodes[:10]  # You can change the number to print more or fewer nodes

for node, closeness in top_nodes:
    print(f"Node {node}: Closeness Centrality: {closeness:.4f}")  
print("============================================")