import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import warnings
import itertools
import plotly.graph_objects as go
import time
from tqdm import tqdm
from collections import defaultdict
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
def p_eccentricity_knn(point_cloud,p,k):

    # Compute the k-NN graph
    nbrs = NearestNeighbors(n_neighbors=k, algorithm='kd_tree').fit(point_cloud)
    _, indices = nbrs.kneighbors(point_cloud)

    # Build the adjacency matrix
    adjacency_matrix = np.zeros((len(point_cloud), len(point_cloud)))
    for i, neighbors in enumerate(indices):
        adjacency_matrix[i, neighbors] = 1
        adjacency_matrix[neighbors, i] = 1

    graph = nx.from_numpy_array(adjacency_matrix)

    D = np.array(nx.floyd_warshall_numpy(graph))

    return np.sum(D**p,axis = 0)**(1/p)

def p_eccentricity_VR(data,p):

    DRG = DecoratedReebGraph(data = data, function = p_eccentricity_knn(data,p,10))
    DRG.fit_Vietoris_Rips()
    D = np.array(nx.floyd_warshall_numpy(DRG.VRGraph))

    return np.sum(D**p,axis = 0)**(1/p)





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


# Preprocess constants
pixel_size = 0.001
birth_range = (0, 0.3)
pers_range = (0, 0.3)
kernel_params = {'sigma': [[0.00001, 0.0], [0.0, 0.00001]]}

# Utility: Calculate eccentricity
def p_eccentricity_knn(point_cloud, p, k):
    # Compute the k-NN graph
    nbrs = NearestNeighbors(n_neighbors=k, algorithm='kd_tree').fit(point_cloud)
    _, indices = nbrs.kneighbors(point_cloud)

    # Build the adjacency matrix
    adjacency_matrix = np.zeros((len(point_cloud), len(point_cloud)))
    for i, neighbors in enumerate(indices):
        adjacency_matrix[i, neighbors] = 1
        adjacency_matrix[neighbors, i] = 1

    graph = nx.from_numpy_array(adjacency_matrix)

    D = np.array(nx.floyd_warshall_numpy(graph))

    return np.sum(D ** p, axis=0) ** (1 / p)


def p_eccentricity_VR(data, p):
    DRG = DecoratedReebGraph(data=data, function=p_eccentricity_knn(data, p, 10))
    DRG.fit_Vietoris_Rips()
    D = np.array(nx.floyd_warshall_numpy(DRG.VRGraph))

    return np.sum(D ** p, axis=0) ** (1 / p)

# Preprocess all point clouds into DRG format
def preprocess_to_DRG(point_clouds, p=100):
    DRG_objects = []
    invalid_indices = []  # List to store indices with warnings
    start_time = time.time()

    for idx, point_cloud in enumerate(tqdm(point_clouds, desc="Preprocessing point clouds to DRGs")):
        try:
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")  # Capture all warnings
                # function = p_eccentricity_VR(point_cloud, p)
                function = point_cloud[:,2]
                DRG = DecoratedReebGraph(data=point_cloud, function=function)
                DRG.fit_Vietoris_Rips()
                DRG.fit_Reeb()
                DRG.fit_diagrams()
                # Check for specific warnings
                for warning in w:
                    if "invalid value encountered in scalar subtract" in str(warning.message):
                        invalid_indices.append(idx)
                        break  # Skip adding the DRG for this index
                else:
                    DRG_objects.append(DRG)  # Add only if no warning
        except Exception as e:
            print(f"Error processing point cloud at index {idx}: {e}")
            invalid_indices.append(idx)

    preprocess_time = time.time() - start_time
    print(f"Preprocessing completed in {preprocess_time:.2f} seconds.")
    print(f"Invalid indices encountered: {invalid_indices}")
    return DRG_objects, preprocess_time, invalid_indices

def avg_sym_reeb_radius(G):
    """
    Compute the symmetric Reeb radius matrix using 0.5 * (p(x, y) + p(y, x)).
    """
    matrix, nodes = compute_reeb_radius_matrix_numpy(G)
    sym_matrix = 0.5 * (matrix + matrix.T)
    return sym_matrix, nodes

def compute_gromov_wasserstein_distance(DRG0, DRG1):
    # Compute the radius matrices and node lists
    reeb_radius_matrix_0, node_list_0 = avg_sym_reeb_radius(DRG0.ReebGraph)
    reeb_radius_matrix_1, node_list_1 = avg_sym_reeb_radius(DRG1.ReebGraph)

    # Get persistence probabilities and node weights
    F_mapped_persistence_pairs, F_node_weights = persistent_probability(DRG0.ReebGraph)
    G_mapped_persistence_pairs, G_node_weights = persistent_probability(DRG1.ReebGraph)

    # Get normalized node weights
    P_R_f = np.array([F_node_weights.get(node, 0) for node in node_list_0])
    P_R_g = np.array([G_node_weights.get(node, 0) for node in node_list_1])

    # Normalize weights
    P_R_f /= np.sum(P_R_f) if np.sum(P_R_f) > 0 else len(P_R_f)
    P_R_g /= np.sum(P_R_g) if np.sum(P_R_g) > 0 else len(P_R_g)

    # Convert cost matrices to numpy arrays
    D_R_f = np.array(reeb_radius_matrix_0)
    D_R_g = np.array(reeb_radius_matrix_1)

    # Compute Gromov-Wasserstein distance
    gw_distance, log = ot.gromov.gromov_wasserstein2(
        D_R_f, D_R_g, P_R_f, P_R_g, 'square_loss', log=True
    )

    return gw_distance


def evaluate_knn(test_DRGs, train_DRGs, train_labels, test_labels, k_values):
    start_time = time.time()

    # Compute all pairwise distances once
    all_distances = []
    print("Calculating pairwise distances between test and training DRGs...")
    for test_DRG in tqdm(test_DRGs, desc="Calculating distances for test DRGs"):
        distances = [
            compute_gromov_wasserstein_distance(test_DRG, train_DRG)
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
    print(f"Evaluation completed in {evaluation_time:.2f} seconds.")
    return accuracies, evaluation_time



# Main execution
if __name__ == "__main__":
    # # Load the dataset
    # with open("Data/SHRECPointClouds1024.pickle", "rb") as f:
    with open("Data/modelNetPointClouds1024.pickle", "rb") as f:
        full_dataset = pickle.load(f)
    # # #

    np.random.seed(42)
    full_dataset = {
        'train_points': [np.random.rand(100, 3) for _ in range(1000)],
        'test_points': [np.random.rand(100, 3) for _ in range(500)],
        'train_labels': np.random.randint(1, 11, 1000),
        'test_labels': np.random.randint(1, 11, 500),
    }

    # Input data
    training_data_point_clouds = np.array(full_dataset['train_points'])
    test_data_point_clouds = np.array(full_dataset['test_points'])
    training_labels = np.array(full_dataset['train_labels'])
    test_labels = np.array(full_dataset['test_labels'])

    # Parameters
    num_train_samples = 400
    num_test_samples = 90


    # Helper function to sample uniformly across labels
    def sample_uniformly(data_points, labels, num_samples):
        unique_labels = np.unique(labels)
        samples_per_label = num_samples // len(unique_labels)
        sampled_points = []
        sampled_labels = []

        for label in unique_labels:
            indices = np.where(labels == label)[0]
            sampled_indices = np.random.choice(indices, samples_per_label, replace=False)
            sampled_points.extend(data_points[sampled_indices])
            sampled_labels.extend(labels[sampled_indices])

        return np.array(sampled_points), np.array(sampled_labels)


    # Sample uniformly for training and testing
    train_points,train_labels = sample_uniformly(
        training_data_point_clouds, training_labels, num_train_samples
    )

    test_points, test_labels = sample_uniformly(
        test_data_point_clouds, test_labels, num_test_samples
    )



    # # # # Step 1: Preprocess training point clouds into DRGs
    pixel_size = 0.001
    birth_range = (0, 0.3)
    pers_range = (0, 0.3)
    kernel_params = {'sigma': [[0.00001, 0.0], [0.0, 0.00001]]}
    # # #
    train_DRGs, train_time, train_invalid = preprocess_to_DRG(train_points)
    test_DRGs, test_time, test_invalid = preprocess_to_DRG(test_points)

    # Remove invalid entries from DRGs and labels
    train_DRGs = [drg for i, drg in enumerate(train_DRGs) if i not in train_invalid]
    train_labels = [l for i, l in enumerate(train_labels) if i not in train_invalid]
    test_DRGs = [drg for i, drg in enumerate(test_DRGs) if i not in test_invalid]
    test_labels = [l for i, l in enumerate(test_labels) if i not in test_invalid]
    # #
    # print("Test data is done.")
    # # Save the results of the first step for reuse
    with open("train_DRGs_model10.pkl", "wb") as f:
        pickle.dump({"DRGs": train_DRGs, "labels": train_labels}, f)
    with open("test_DRGs_model10.pkl", "wb") as f:
        pickle.dump({"DRGs": test_DRGs, "labels": test_labels}, f)

    # print("Saved preprocessed train and test DRGs to disk.")


    # Verify the loaded data
    print(f"Loaded {len(train_DRGs)} train DRGs and {len(train_labels)} train labels.")
    print(f"Loaded {len(test_DRGs)} test DRGs and {len(test_labels)} test labels.")

    # Step 2: Evaluate k-NN accuracy
    k_values = [1,3,5,10, 15, 20,25,30,35,40]
    accuracies, eval_time = evaluate_knn(test_DRGs, train_DRGs, train_labels, test_labels, k_values)

    # Final Output
    # print(f"Preprocessing time (training): {train_time:.2f} seconds.")
    # print(f"Preprocessing time (testing): {test_time:.2f} seconds.")
    print(f"Evaluation time: {eval_time:.2f} seconds.")
    print("Accuracies:")
    for k, accuracy in accuracies.items():
        print(f"  k = {k}: {accuracy:.2%}")
    print()