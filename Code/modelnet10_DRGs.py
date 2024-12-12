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
import os
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


def load_obj_files_and_labels(root_dir):
    data = []  # To store point clouds
    labels = []  # To store corresponding labels

    for folder_name in os.listdir(root_dir):
        folder_path = os.path.join(root_dir, folder_name)

        if os.path.isdir(folder_path):
            label = folder_name  # Use folder name as label

            for file_name in os.listdir(folder_path):
                if file_name.endswith(".obj"):
                    file_path = os.path.join(folder_path, file_name)
                    try:
                        # Load the mesh
                        mesh = trimesh.load(file_path)

                        # Check for invalid vertices or faces
                        if mesh.vertices.shape[0] == 0 or mesh.faces.shape[0] == 0:
                            print(f"Invalid geometry in {file_path}. Skipping.")
                            continue
                        if np.isnan(mesh.vertices).any() or np.isinf(mesh.vertices).any():
                            print(f"Invalid vertex data in {file_path}. Skipping.")
                            continue

                        # Update faces to remove degenerate ones
                        valid_faces = mesh.nondegenerate_faces()
                        mesh.update_faces(valid_faces)

                        # Remove unreferenced vertices
                        mesh.remove_unreferenced_vertices()

                        # Close small holes if possible
                        mesh.fill_holes()

                        # Sample point cloud
                        point_cloud = mesh.sample(1024)

                        # Store the point cloud and label
                        data.append(point_cloud)
                        labels.append(label)
                    except Exception as e:
                        print(f"Error processing {file_path}: {e}")
                        continue

    return data, labels

def p_eccentricity_VR(data, p):
    DRG = DecoratedReebGraph(data=data, function=p_eccentricity_knn(data, p, 10))
    DRG.fit_Vietoris_Rips()
    D = np.array(nx.floyd_warshall_numpy(DRG.VRGraph))

    return np.sum(D ** p, axis=0) ** (1 / p)

def preprocess_to_DRG(point_clouds, p=100):
    DRG_objects = []
    invalid_indices = []  # List to store indices with warnings
    start_time = time.time()

    for idx, point_cloud in enumerate(tqdm(point_clouds, desc="Preprocessing point clouds to DRGs")):
        try:
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")  # Capture all warnings
                function = p_eccentricity_VR(point_cloud, p)
                DRG = DecoratedReebGraph(data=point_cloud, function=function)
                DRG.fit_Vietoris_Rips()
                DRG.fit_Reeb()
                DRG.fit_diagrams(
                    persistence_images=True,
                    birth_range=(0, 0.3),
                    pixel_size=0.001,
                    kernel_params={'sigma': [[0.00001, 0.0], [0.0, 0.00001]]},
                )
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



# k-NN classification and accuracy evaluation
def evaluate_knn(test_DRGs, train_DRGs, train_labels, test_labels, k_values):
    start_time = time.time()

    # Compute all pairwise distances once
    all_distances = []
    print("Calculating pairwise distances between test and training DRGs...")
    for test_DRG in tqdm(test_DRGs, desc="Calculating distances for test DRGs"):
        distances = [
            DRG_distance(test_DRG.ReebGraph, train_DRG.ReebGraph, attribute="diagram")
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


import numpy as np
import random
import pickle
import warnings
from tqdm import tqdm
import time


def preprocess_to_DRG(point_clouds, p=100):
    DRG_objects = []
    invalid_indices = []  # List to store indices with warnings
    start_time = time.time()
    for idx, point_cloud in enumerate(tqdm(point_clouds, desc="Preprocessing point clouds to DRGs")):
        try:
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")  # Capture all warnings
                function = p_eccentricity_VR(point_cloud, p)
                DRG = DecoratedReebGraph(data=point_cloud, function=function)
                DRG.fit_Vietoris_Rips()
                DRG.fit_Reeb()
                DRG.fit_diagrams(
                    persistence_images=True,
                    birth_range=(0, 0.3),
                    pixel_size=0.001,
                    kernel_params={'sigma': [[0.00001, 0.0], [0.0, 0.00001]]},
                )
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


if __name__ == "__main__":
    train_file = "train_DRGs_model10.pkl"
    test_file = "test_DRGs_model10.pkl"
    #
    # Load train DRGs and labels
    with open(train_file, "rb") as f:
        train_data = pickle.load(f)
        train_DRGs = train_data["DRGs"]
        train_labels = train_data["labels"]

    print("Load training data is done.")

    with open(test_file, "rb") as f:
        test_data = pickle.load(f)
        test_DRGs = test_data["DRGs"]
        test_labels = test_data["labels"]

    # Verify the loaded data
    print(f"Loaded {len(train_DRGs)} train DRGs and {len(train_labels)} train labels.")
    print(f"Loaded {len(test_DRGs)} test DRGs and {len(test_labels)} test labels.")

    # Step 2: Evaluate k-NN accuracy
    k_values = [1, 3, 5, 10, 15, 20, 25, 30, 35, 40]
    accuracies, eval_time = evaluate_knn(test_DRGs, train_DRGs, train_labels, test_labels, k_values)


    print(f"Evaluation time: {eval_time:.2f} seconds.")
    print("Accuracies:")
    for k, accuracy in accuracies.items():
        print(f"  k = {k}: {accuracy:.2%}")
    print()