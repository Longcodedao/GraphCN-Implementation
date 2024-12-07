from dataset.deepwalk_dataset import *
from model.DeepWalk import DeepWalk, DeepWalk_HierSoftmax
from utils.deepwalk_utils import *

import pandas as pd
from torch.utils.data import Dataset
import networkx as nx
from tqdm import tqdm

import numpy as np
import networkx as nx
import threading
import torch.nn as nn
import torch.optim as optim
import torch
from torch.utils.data import Dataset, DataLoader

import matplotlib.pyplot as plt
from concurrent.futures import ThreadPoolExecutor

from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import hamming_loss, f1_score




def learning_embed(training_dataset, model, num_epochs):
    vertex2idx = training_dataset.vertex_to_index
    idx2vertex = training_dataset.index_to_vertex
    num_nodes = len(list(vertex2idx.values()))

    # print(num_nodes)
    training_loader_embed = DataLoader(training_dataset, batch_size = 32, shuffle = True)
    deepwalk = model(num_nodes, 128).to(device)
    optimizer = optim.Adam(deepwalk.parameters(), lr = 0.001)
    
    deepwalk.train()
    for epoch in range(num_epochs):
        total_loss = 0
        for center, context in training_loader:
            optimizer.zero_grad()
            center = center.to(device)
            context = context.to(device)

            neg_logp = deepwalk(center, context)
    
            avg_logp = neg_logp.mean()
            avg_logp.backward()
            optimizer.step()
            total_loss += avg_logp
        
        print(f'Epoch: {epoch + 1}, Loss: {total_loss / len(training_loader)}')

    return deepwalk



def node_classification(model, training_data, labels_df, labels_node_df):
    model.eval()

    idx2vertex = training_data.index_to_vertex
    vertex2idx = training_data.vertex_to_index
    # print(len(vertex2idx))
    
    # Creating the dataset for node_classification
    # Take out the embeddings after training to Numpy
    node_embeddings = model.embeddings.weight.detach().cpu().numpy()
    # print(node_embeddings.shape)
    labels = label_nodes_df.groupby('node')['label'].apply(list).to_dict()
    # print(labels[1762])
    labels_scope = {vertex: labels[vertex] for vertex in labels if vertex in vertex2idx}

    # print(labels_scope[1762])
    
    # Index to label 
    idx2label = {}
    for idx, label in enumerate(labels_df['label'].unique()):
        idx2label[idx] = label

    # print(idx2label)
    
    label2idx = {label: idx for idx, label in idx2label.items()} 
    num_labels = len(list(label2idx.keys()))

    num_nodes = len(list(labels_scope.keys()))
    
    Y = np.zeros((num_nodes, num_labels))
    for idx, node in idx2vertex.items():
        # print(idx, node)
        if node in labels_scope:
            for label in labels_scope.get(node, []):  
                index_label = label2idx[label]
                Y[idx, index_label] = 1

    # print(Y[1762, :])
    X_train, X_test, y_train, y_test = train_test_split(node_embeddings, Y, 
                                                        test_size = 0.5, 
                                                        random_state = 42)

    
    # Train a Liblinear model for multi-label classification (binary relevance approach)
    # We'll use one Logistic Regression per label
    clf = OneVsRestClassifier(
        LogisticRegression(
            solver = "liblinear",
            max_iter = 100)
    )
        
    clf.fit(X_train, y_train)

    y_score = clf.predict_proba(X_test) 
    num_label = np.sum(y_test, axis = 1, dtype = np.int64)
    # Take the largest first and then descending
    y_sort = np.fliplr(np.argsort(y_score, axis = 1))
    y_pred = np.zeros_like(y_test, dtype=np.int64)
    
    for i in range(y_test.shape[0]):
        for j in range(num_label[i]):
            y_pred[i, y_sort[i, j]] = 1

    # Evaluate the model
    hamming = hamming_loss(y_test, y_pred)
    # print(f'Test:\n', y_test)
    # print(f'Predict:\n', y_pred)
    f1_macro = f1_score(y_test, y_pred, average='macro', zero_division=0)
    f1_micro = f1_score(y_test, y_pred, average='micro', zero_division=0)

    print(f"Hamming Loss: {hamming}")
    print(f"Macro F1 Score: {f1_macro}")
    print(f"Micro F1 Score: {f1_micro}")
    
    return hamming, f1_macro, f1_micro

def main():

    num_walks = 30
    walk_length = 8
    window_size = 5
    
    nodes_df = pd.read_csv('BlogCatalog-dataset/data/nodes.csv',
                   header = None, names = ['id'])
    edges_df = pd.read_csv('BlogCatalog-dataset/data/edges.csv',
                       header = None, names = ['source', 'target'])
    labels_df = pd.read_csv('BlogCatalog-dataset/data/groups.csv',
                        header = None, names = ['label'])
    labels_node_df = pd.read_csv('BlogCatalog-dataset/data/group-edges.csv',
                            header = None, names = ['node', 'label'])


    G = nx.Graph()
    
    for _, row in nodes_df.iterrows():
        G.add_node(row['id'])
    
    G.add_edges_from(edges_df.values)


    training_data = random_walks_dataset(G, num_walks, walk_length, window_size)
    embed_model = learning_embed(training_data, DeepWalk_HierSoftmax, num_epochs)

    hamming, f1_macro, f1_micro = node_classification(embed_model_lol, 
                                                                  training_data, 
                                                  labels_df, labels_node_df)
    

if __name__ == '__main__':
    main()
    