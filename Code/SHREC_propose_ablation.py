import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import warnings
import itertools
import plotly.graph_objects as go
import time
from tqdm import tqdm

from gtda.plotting import plot_point_cloud
from ripser import ripser
from persim import PersistenceImager, plot_diagrams
import gudhi as gd
from ot import fused_gromov_wasserstein2
import ot

import trimesh

from DecoratedReebGraphs import *
import pickle
from sklearn.neighbors import NearestNeighbors
import warnings

# # Suppress the specific warning from ripser
warnings.filterwarnings("ignore", message="The input point cloud has more columns than rows; did you mean to transpose?")
warnings.filterwarnings(
    "ignore",
    message="The input matrix is square, but the distance_matrix flag is off.  Did you mean to indicate that this was a distance matrix?",
)

# warnings.filterwarnings(
#     "ignore",
#     message="invalid value encountered in scalar subtract",
# )

import numpy as np
import networkx as nx
import heapq

def reeb_radius(G, root):
    """
    Compute the Reeb radius for a given root node in the modified graph G.
    """
    # Initialize Reeb radius dictionary
    rho_g = {v: float('inf') for v in G.nodes}
    rho_g[root] = 0

    # Priority queue for Dijkstra-style processing
    Q = [(0, root)]  # (Reeb radius, node)

    # Helper function to get the node value
    def get_node_value(node):
        if 'f_vertex' in G.nodes[node]:
            return G.nodes[node]['f_vertex']
        elif 'function value' in G.nodes[node]:
            return G.nodes[node]['function value']
        else:
            raise KeyError(f"Node {node} does not have 'f_vertex' or 'function value'")

    # Extract root's value for easier comparison
    root_value = get_node_value(root)

    while Q:
        current_radius, v = heapq.heappop(Q)

        for neighbor in G.neighbors(v):
            # Get the values for the current node and its neighbor
            v_value = get_node_value(v)
            neighbor_value = get_node_value(neighbor)

            # Compute the distance from the root to the neighbor
            d_m = abs(root_value - neighbor_value)

            # Compute the updated radius for the neighbor
            updated_radius = max(rho_g[v], d_m)

            # Ensure paths respect the supremum logic
            if root_value >= max(v_value, neighbor_value):
                updated_radius = 0

            # Update radius and priority queue if a better path is found
            if updated_radius < rho_g[neighbor]:
                rho_g[neighbor] = updated_radius
                heapq.heappush(Q, (updated_radius, neighbor))

    return rho_g

def compute_reeb_radius_matrix_numpy(G):
    """
    Compute the Reeb radius matrix for the entire graph G using numpy.
    """
    nodes = list(G.nodes)
    n = len(nodes)
    node_index_map = {node: idx for idx, node in enumerate(nodes)}  # Map node to index for matrix operations

    # Initialize the matrix with infinities
    matrix = np.full((n, n), float('inf'))

    for i, node in enumerate(nodes):
        radii = reeb_radius(G, node)
        for other_node, radius in radii.items():
            j = node_index_map[other_node]
            matrix[i, j] = radius

    return matrix, nodes  # Return the matrix and the node list for reference

# Example Usage with the Modified Graph
# Assume G is constructed using `Reeb_approx_graph`

# Calculate the Reeb radius matrix


def persistent_probability(graph):
    """
    Compute persistent probabilities for a graph with nodes and edges,
    where nodes may have custom attributes such as 'function value'.
    """

    # Step 1: Create a mapping between node names (tuples) and integers
    node_mapping = {node: idx for idx, node in enumerate(graph.nodes)}
    inverse_node_mapping = {idx: node for node, idx in node_mapping.items()}

    # Relabel graph nodes for SimplexTree processing
    mapped_graph = nx.relabel_nodes(graph, node_mapping)

    # Step 2: Build the SimplexTree
    st = gd.SimplexTree()

    # Add nodes with filtration values (assumes 'function value' attribute)
    for node, attr in mapped_graph.nodes(data=True):
        if 'function value' not in attr:
            raise ValueError(f"Node {node} missing 'function value' attribute.")
        st.insert([node], filtration=attr['function value'])

    # Add edges with filtration values
    for u, v in mapped_graph.edges:
        filtration_value = max(mapped_graph.nodes[u]['function value'],
                               mapped_graph.nodes[v]['function value'])
        st.insert([u, v], filtration=filtration_value)

    # Step 3: Make filtration non-decreasing and compute extended persistence
    st.make_filtration_non_decreasing()
    st.extend_filtration()
    extended_persistence = st.extended_persistence()

    # Step 4: Map persistence diagrams to node pairs
    node_filtrations = {
        inverse_node_mapping[idx]: attr['function value']
        for idx, attr in mapped_graph.nodes(data=True)
    }

    mapped_persistence_pairs = {}
    diagram_types = ["Ordinary", "Relative", "Extended+", "Extended-"]
    for diagram_type, dgm in zip(diagram_types, extended_persistence):
        mapped_persistence_pairs[diagram_type] = []
        for interval in dgm:
            birth, death = interval[1]
            birth_nodes = [
                node
                for node, filtration in node_filtrations.items()
                if abs(filtration - birth) < 1e-6
            ]
            death_nodes = [
                node
                for node, filtration in node_filtrations.items()
                if abs(filtration - death) < 1e-6
            ]

            if birth_nodes and death_nodes:
                mapped_persistence_pairs[diagram_type].append((birth_nodes[0], death_nodes[0]))
            elif birth_nodes:
                mapped_persistence_pairs[diagram_type].append((birth_nodes[0], None))
            elif death_nodes:
                mapped_persistence_pairs[diagram_type].append((None, death_nodes[0]))

    # Step 5: Calculate lifespan and normalize
    lifespans = {}
    for diagram_type, pairs in mapped_persistence_pairs.items():
        for birth, death in pairs:
            if birth is not None and death is not None:
                lifespan = abs(node_filtrations[birth] - node_filtrations[death])
                lifespans[(birth, death)] = lifespan

    total_lifespan = sum(lifespans.values())
    normalized_lifespans_sum = {
        pair: lifespan / total_lifespan if total_lifespan != 0 else 0
        for pair, lifespan in lifespans.items()
    }

    # Step 6: Recalculate node weights
    node_weights = {}
    for (node1, node2), normalized_lifespan in normalized_lifespans_sum.items():
        weight = normalized_lifespan / 2
        if node1 is not None:
            node_weights[node1] = node_weights.get(node1, 0) + weight
        if node2 is not None:
            node_weights[node2] = node_weights.get(node2, 0) + weight

    return mapped_persistence_pairs, node_weights
# 1. Maximum Symmetric Reeb Radius
def max_sym_reeb_radius(G):
    """
    Compute the symmetric Reeb radius matrix using max(p(x, y), p(y, x)).
    """
    matrix, nodes = compute_reeb_radius_matrix_numpy(G)
    sym_matrix = np.maximum(matrix, matrix.T)
    return sym_matrix, nodes

# 2. Average Symmetric Reeb Radius
def avg_sym_reeb_radius(G):
    """
    Compute the symmetric Reeb radius matrix using 0.5 * (p(x, y) + p(y, x)).
    """
    matrix, nodes = compute_reeb_radius_matrix_numpy(G)
    sym_matrix = 0.5 * (matrix + matrix.T)
    return sym_matrix, nodes

# 3. Shortest Path Distance
def sp_reeb_radius(G):
    """
    Compute the shortest path distance matrix for a graph G.
    """
    # Use NetworkX's shortest path length function
    sp_lengths = dict(nx.all_pairs_dijkstra_path_length(G, weight='weight'))
    nodes = list(G.nodes)
    node_index_map = {node: idx for idx, node in enumerate(nodes)}  # Map node to index
    n = len(nodes)

    # Initialize the matrix
    sp_matrix = np.full((n, n), float('inf'))
    for i, node1 in enumerate(nodes):
        for node2, length in sp_lengths[node1].items():
            j = node_index_map[node2]
            sp_matrix[i, j] = length

    return sp_matrix, nodes

# 4. Reeb Distance
def reeb_distance(G):
    """
    Compute the Reeb distance matrix for the graph G.

    The Reeb distance is defined as the difference between the maximum and minimum
    scalar values along the shortest path between two nodes.

    Parameters:
        G (networkx.Graph): The Reeb graph with nodes having a 'function value' attribute.

    Returns:
        distance_matrix (np.ndarray): A symmetric distance matrix of Reeb distances.
        node_list (list): List of nodes corresponding to the rows/columns of the matrix.
    """
    nodes = list(G.nodes)
    n = len(nodes)
    node_index_map = {node: idx for idx, node in enumerate(nodes)}

    # Initialize the matrix
    distance_matrix = np.zeros((n, n))

    # Precompute shortest paths
    shortest_paths = dict(nx.all_pairs_shortest_path(G))

    for i, source in enumerate(nodes):
        for j, target in enumerate(nodes):
            if i == j:
                # Reeb distance from a node to itself is 0
                distance_matrix[i, j] = 0
            else:
                # Get the shortest path between source and target
                path = shortest_paths[source][target]

                # Extract the 'function value' for each node in the path
                path_values = [G.nodes[node].get('function value', 0) for node in path]

                # Compute the Reeb distance: max value - min value on the path
                distance_matrix[i, j] = max(path_values) - min(path_values)

    return distance_matrix, nodes



def compute_gromov_wasserstein_distance(DRG0, DRG1, dist="reeb_radius", weight="uniform"):
    # Compute the distance matrices and node lists based on the selected distance type
    if dist == "reeb_radius":
        reeb_radius_matrix_0, node_list_0 = compute_reeb_radius_matrix_numpy(DRG0.ReebGraph)
        reeb_radius_matrix_1, node_list_1 = compute_reeb_radius_matrix_numpy(DRG1.ReebGraph)
    elif dist == "max_sym_reeb_radius":
        reeb_radius_matrix_0, node_list_0 = max_sym_reeb_radius(DRG0.ReebGraph)
        reeb_radius_matrix_1, node_list_1 = max_sym_reeb_radius(DRG1.ReebGraph)
    elif dist == "avg_sym_reeb_radius":
        reeb_radius_matrix_0, node_list_0 = avg_sym_reeb_radius(DRG0.ReebGraph)
        reeb_radius_matrix_1, node_list_1 = avg_sym_reeb_radius(DRG1.ReebGraph)
    elif dist == "sp_reeb_radius":
        reeb_radius_matrix_0, node_list_0 = sp_reeb_radius(DRG0.ReebGraph)
        reeb_radius_matrix_1, node_list_1 = sp_reeb_radius(DRG1.ReebGraph)
    elif dist == "reeb_distance":
        reeb_radius_matrix_0, node_list_0 = reeb_distance(DRG0.ReebGraph)
        reeb_radius_matrix_1, node_list_1 = reeb_distance(DRG1.ReebGraph)
    else:
        raise ValueError(f"Unknown distance type: {dist}")

    # Compute node weights based on the selected weight type
    if weight == "uniform":
        P_R_f = np.ones(len(node_list_0)) / len(node_list_0)
        P_R_g = np.ones(len(node_list_1)) / len(node_list_1)
    elif weight == "function_value":
        # Directly use function values as weights
        P_R_f = np.array([DRG0.ReebGraph.nodes[node].get('function value', 0) for node in node_list_0])
        P_R_g = np.array([DRG1.ReebGraph.nodes[node].get('function value', 0) for node in node_list_1])
        P_R_f /= np.sum(P_R_f) if np.sum(P_R_f) > 0 else len(P_R_f)
        P_R_g /= np.sum(P_R_g) if np.sum(P_R_g) > 0 else len(P_R_g)
    elif weight == "degree_based":
        P_R_f = np.array([DRG0.ReebGraph.degree(node) for node in node_list_0], dtype=float)  # Convert to float
        P_R_g = np.array([DRG1.ReebGraph.degree(node) for node in node_list_1], dtype=float)  # Convert to float
        P_R_f /= np.sum(P_R_f) if np.sum(P_R_f) > 0 else len(P_R_f)
        P_R_g /= np.sum(P_R_g) if np.sum(P_R_g) > 0 else len(P_R_g)
    elif weight == "persist":
        # Use persistence-based probabilities as weights
        F_mapped_persistence_pairs, F_node_weights = persistent_probability(DRG0.ReebGraph)
        G_mapped_persistence_pairs, G_node_weights = persistent_probability(DRG1.ReebGraph)
        P_R_f = np.array([F_node_weights.get(node, 0) for node in node_list_0])
        P_R_g = np.array([G_node_weights.get(node, 0) for node in node_list_1])
        P_R_f /= np.sum(P_R_f) if np.sum(P_R_f) > 0 else len(P_R_f)
        P_R_g /= np.sum(P_R_g) if np.sum(P_R_g) > 0 else len(P_R_g)
    else:
        raise ValueError(f"Unknown weight type: {weight}")

    # Convert cost matrices to numpy arrays
    D_R_f = np.array(reeb_radius_matrix_0)
    D_R_g = np.array(reeb_radius_matrix_1)

    # Compute Gromov-Wasserstein distance
    gw_distance, log = ot.gromov.gromov_wasserstein2(
        D_R_f, D_R_g, P_R_f, P_R_g, 'square_loss', log=True
    )

    return gw_distance


def evaluate_knn(test_DRGs, train_DRGs, train_labels, test_labels, k_values, distance_types):
    """
    Evaluate k-NN classification for multiple distance types.

    Parameters:
        test_DRGs: list of test DRG objects.
        train_DRGs: list of training DRG objects.
        train_labels: list of training labels.
        test_labels: list of test labels.
        k_values: list of k values for k-NN.
        distance_types: list of distance types to evaluate.

    Returns:
        results: Dictionary with distance types as keys and accuracies as values.
    """
    results = {}

    for dist_type in distance_types:
        print(f"\nEvaluating using distance type: {dist_type}")
        start_time = time.time()

        # Compute all pairwise distances for the current distance type
        all_distances = []
        print("Calculating pairwise distances between test and training DRGs...")
        for test_DRG in tqdm(test_DRGs, desc=f"Calculating distances for test DRGs ({dist_type})"):
            distances = [
                compute_gromov_wasserstein_distance(test_DRG, train_DRG, dist=dist_type)
                for train_DRG in train_DRGs
            ]
            all_distances.append(np.argsort(distances))  # Store sorted indices for each test DRG

        # Calculate accuracies for each k
        accuracies = {}
        for k in k_values:
            correct_predictions = 0
            for test_idx, sorted_indices in enumerate(all_distances):
                # Get k closest neighbors
                k_closest_indices = sorted_indices[:k]
                # Get the labels of k closest neighbors
                k_closest_labels = [train_labels[idx] for idx in k_closest_indices]
                # Check if the true label is in the predicted labels
                if test_labels[test_idx] in k_closest_labels:
                    correct_predictions += 1
            # Compute accuracy for this k
            accuracies[k] = correct_predictions / len(test_DRGs)

        evaluation_time = time.time() - start_time
        results[dist_type] = (accuracies, evaluation_time)

        # Print results immediately
        print(f"Results for distance type: {dist_type}")
        print(f"Evaluation time: {evaluation_time:.2f} seconds.")
        print("Accuracies:")
        for k, accuracy in accuracies.items():
            print(f"  k = {k}: {accuracy:.2%}")
        print()

    return results

# k-NN Evaluation
def evaluate_knn_with_fixed_distance(test_DRGs, train_DRGs, train_labels, test_labels, k_values, weight_types):
    results = {}
    distance_type = "avg_sym_reeb_radius"
    print(f"Evaluating with fixed distance type: {distance_type}")

    for weight_type in weight_types:
        print(f"\nUsing weight type: {weight_type}")
        start_time = time.time()
        all_distances = []

        for test_DRG in tqdm(test_DRGs, desc=f"Calculating distances ({weight_type})"):
            distances = [
                compute_gromov_wasserstein_distance(test_DRG, train_DRG, dist=distance_type, weight=weight_type)
                for train_DRG in train_DRGs
            ]
            all_distances.append(np.argsort(distances))

        accuracies = {}
        for k in k_values:
            correct_predictions = 0
            for test_idx, sorted_indices in enumerate(all_distances):
                k_closest_indices = sorted_indices[:k]
                k_closest_labels = [train_labels[idx] for idx in k_closest_indices]
                if test_labels[test_idx] in k_closest_labels:
                    correct_predictions += 1
            accuracies[k] = correct_predictions / len(test_DRGs)

        evaluation_time = time.time() - start_time
        results[weight_type] = (accuracies, evaluation_time)

        print(f"\nResults for weight type: {weight_type}")
        print(f"Evaluation time: {evaluation_time:.2f} seconds.")
        for k, accuracy in accuracies.items():
            print(f"  k = {k}: {accuracy:.2%}")
    return results

# Main execution
if __name__ == "__main__":

    train_file = "train_DRGs_shrec.pkl"
    test_file = "test_DRGs_shrec.pkl"
#
    #Load train DRGs and labels
    with open(train_file, "rb") as f:
        train_data = pickle.load(f)
        train_DRGs = train_data["DRGs"]
        train_labels = train_data["labels"]
        manual_invalid = [69, 121, 169, 171, 222, 243, 283, 299]
        train_labels = [l for i, l in enumerate(train_labels) if i not in manual_invalid]
    print("Load training data is done.")

    with open(test_file, "rb") as f:
        test_data = pickle.load(f)
        test_DRGs = test_data["DRGs"]
        test_labels = test_data["labels"]
    # Load test DRGs and labels
    # with open(test_file, "rb") as f:
    #     test_data = pickle.load(f)
    #     test_DRGs = test_data["DRGs"]
    #     test_labels = test_data["labels"]

    # Verify the loaded data
    print(f"Loaded {len(train_DRGs)} train DRGs and {len(train_labels)} train labels.")
    print(f"Loaded {len(test_DRGs)} test DRGs and {len(test_labels)} test labels.")

    # Step 2: Evaluate k-NN accuracy

    # Distance types to evaluate
    # distance_types = ["max_sym_reeb_radius", "avg_sym_reeb_radius", "sp_reeb_radius", "reeb_distance"]

    # Evaluate k-NN accuracy
    # k_values = [10, 15, 20, 25, 30, 35, 40]
    # results = evaluate_knn(test_DRGs, train_DRGs, train_labels, test_labels, k_values, distance_types)
    # weight_types = ["uniform", "function_value", "degree_based"]
    weight_types = ["degree_based"]
    k_values = [10, 15, 20, 25, 30,35,40]

    evaluate_knn_with_fixed_distance(test_DRGs, train_DRGs, train_labels, test_labels, k_values, weight_types)
