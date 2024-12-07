from torch.utils.data import Dataset, DataLoader


class DeepWalkDataset(Dataset):
    def __init__(self, training_data, vertex_to_index):
        self.training_data = training_data
        self.vertex_to_index = vertex_to_index

        self.index_to_vertex = {index: vertex for vertex, index in self.vertex_to_index.items()}
    def __len__(self):
        return len(self.training_data)

    def __getitem__(self, index):
        center_vertex, context_vertices = self.training_data[index]
        center_index = torch.tensor(self.vertex_to_index[center_vertex])
        context_indices = torch.tensor(self.vertex_to_index[context_vertices])

        return center_index, context_indices