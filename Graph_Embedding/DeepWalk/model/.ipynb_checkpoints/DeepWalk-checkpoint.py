import numpy as np
import networkx as nx
import threading
import torch.nn as nn
import torch.optim as optim
import torch
from torch.utils.data import Dataset, DataLoader

import matplotlib.pyplot as plt
from concurrent.futures import ThreadPoolExecutor

# Initial Deep Walk implementation
class DeepWalk(nn.Module):
    def __init__(self, num_nodes, embedding_dim):
        super(DeepWalk, self).__init__()
        self.embeddings = nn.Embedding(num_nodes, embedding_dim)
        # self.output_layer = nn.Linear(embedding_dim, num_nodes, bias = False)

    def forward(self, center, context):
        # return self.output_layer(self.embeddings(input_walks))

        center_embed = self.embeddings(center)
        context_embed = self.embeddings(context)

        # Applying log-sum-exp trick to solve the overflow problem
        all_logits = torch.matmul(center_embed, self.embeddings.weight.T) # (B, Vocab_size)

  
        result = torch.sum(center_embed * context_embed, dim = 1, keepdim = True)
        
     
        print(all_logits.shape)
        max_logits, _ = torch.max(all_logits, dim = 1, keepdim = True)
        log_sum_exp = max_logits + torch.log(torch.sum(torch.exp(all_logits - max_logits), dim=1, keepdim=True))

        log_prob = result - log_sum_exp

        return -log_prob

        
# De
class DeepWalk_HierSoftmax(nn.Module):
    def __init__(self, num_nodes, embedding_dim):
        super(DeepWalk_HierSoftmax, self).__init__()
        self.num_nodes = num_nodes
        self.embeddings = nn.Embedding(num_nodes, embedding_dim)
        self.probs_tensor = nn.Parameter(torch.rand(2 * num_nodes, embedding_dim), 
                                         requires_grad = True)

    def length_binary_tree_tensors(self, nodes):
        count = torch.zeros_like(nodes)
        while torch.any(nodes > 1):
            nodes = nodes // 2
            count += (nodes > 0).int()
    
        return count

    def path_binary_tree_tensors(self, nodes):
        max_steps = torch.floor(torch.log2(nodes.float())).int() + 1  # Maximum path length for each node
        batch_size = len(nodes)
        
        paths = torch.zeros((batch_size, max_steps.max()), device=nodes.device, dtype=torch.int)
        
        for i in range(max_steps.max()):
            paths[:, i] = nodes
            nodes = nodes // 2  # Update nodes by dividing by 2
            nodes[nodes < 1] = 0  # Stop updating for nodes <= 1
        
        # Remove trailing zeros for each row
        path_lengths = max_steps
        # print(paths)
        trimmed_paths = torch.tensor([paths[i, :path_lengths[i]].tolist() for i in range(batch_size)])
        
        return trimmed_paths


    def forward(self, center, context):
        embed_center = self.embeddings(center)
        # print(f'shape of embed_center : {embed_center.shape}')
        context_node = self.num_nodes + context
        
        length_binary = self.length_binary_tree_tensors(context_node)
        path_bin_tree = self.path_binary_tree_tensors(context_node)

        # Prepare paths as a padded tensor
        max_path_len = length_binary.max()
    
        # print(max_path_len)
        logp = torch.zeros((len(context_node), 1), device=context_node.device)
        for i in range(1, max_path_len):
            # Extract the current level's nodes
            nodes = path_bin_tree[:, i]
    
            # Compute signs based on node parity
            signs = torch.where(nodes % 2 == 0, 1.0, -1.0).unsqueeze(1).to(context_node.device)

            # print(torch.matmul(self.probs_tensor[nodes], embed_center.T).shape)

            dot_product = torch.sum(self.probs_tensor[nodes] * embed_center, dim = 1, keepdim = True)

            probs = torch.sigmoid(signs * dot_product)
            
            # probs = torch.sigmoid(signs * torch.matmul(self.probs_tensor[nodes], embed_center.T))
            # print(probs)
            log_probs = torch.log(probs)
            logp += log_probs  # Aggregate probabilities for the batch
            
        return -logp
