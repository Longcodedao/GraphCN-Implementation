import networkx as nx
import threading
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm



def create_random_walk(graph, start_vertex, length = 5):

    walk = [start_vertex]
    current_vertex = start_vertex
    for _ in range(length):
        neighbors = list(graph.neighbors(current_vertex))
        if not neighbors:  # Handle case where there are no neighbors
            break
        chosen_node = np.random.choice(neighbors)
        walk.append(chosen_node)
        current_vertex = chosen_node

    return walk

# Conducting the algorithm with multi-threading to improve the performance of the model
def random_walk_multi(graph, start_vertices, num_steps, num_threads = 4):
    results = {}

    def worker(start_vertex):
        path = create_random_walk(graph, start_vertex, num_steps)
        results[start_vertex] = path

    with ThreadPoolExecutor(max_workers = num_threads) as executor:
        executor.map(worker, start_vertices)

    return results


# This function creates a Random Walk Dataset
def random_walks_dataset(graph, num_walks, walk_length, window_size):
    # Generate Random Walks part
    all_walks = []
    list_vertices = list(graph.nodes)
    
    for i in tqdm(range(num_walks), desc="Generating Random Walks"):
        walks = random_walk_multi(graph, list_vertices, walk_length, num_threads = 8)
        all_walks.extend(list(walks.values()))

    vertex_to_index = {vertex: index for index, vertex in enumerate(list_vertices)}
    vocab_size = len(vertex_to_index)

    # Skip Gram Part
    training_data = []
    for walk in tqdm(all_walks, desc="Processing walks For Skipgram", unit="walk"):
        for i in range(len(walk)):
            center_vertex = walk[i]

            left_start = max(0, i - window_size)
            right_end = min(len(walk), i + window_size + 1)
            for j in range(left_start, right_end):
                if i != j:
                    training_data.append((center_vertex, walk[j]))

    training_data = DeepWalkDataset(training_data, vertex_to_index)

    return training_data



