'''GNN Architecture with einops.'''

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam

import einops


class GraphConvolution(nn.Module):
    """Graph Convolutional Layer."""
    
    def __init__(self, in_features, out_features):
        super(GraphConvolution, self).__init__()
        self.weight = nn.Parameter(torch.Tensor(in_features, out_features))
        self.bias = nn.Parameter(torch.FloatTensor(out_features))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight)
        nn.init.zeros_(self.bias)

    def forward(self, x, adj):
        """
        Args:
            x: Input features (shape: (num_nodes, in_features))
            adj: Adjacency matrix (shape: (num_nodes, num_nodes))

        This rule captures how the features are updated from one layer to the next,
        considering both the node's features and the features of its neighbors based
        on the normalized adjacency matrix.  It normalizes the adjacency matrix,
        which is crucial for effective message passing between nodes in the graph.
        """

        # Normalize adjacency matrix :  layer-wise propagation rule
        adj = adj + torch.eye(adj.size(0), device=adj.device)  # Add self-loops

        deg = adj.sum(1)

        deg = einops.reduce(adj, 'i j -> i', 'sum') # Calculate degree

        deg_inv_sqrt = deg.pow(-0.5) # Calculate inverse square root of degree: ensure that the influence of each node's neighbors is balanced
        deg_inv_sqrt = einops.rearrange(deg_inv_sqrt, 'i -> i 1') # Reshape deg_inv_sqrt to have shape (N, 1)
        adj_normalized = einops.einsum(deg_inv_sqrt, adj, deg_inv_sqrt, 'i a, i j, j a -> i j')

        # Graph convolution operation using Einops
        x = einops.einsum(adj_normalized, x, 'n i, i j -> n j')  # Propagate features 
        x = einops.einsum(x, self.weight, 'n m, m j -> n j') # Apply weights

        x = F.relu(x)

        return x + self.bias

class GCN(nn.Module):
    """Graph Convolutional Network."""
    def __init__(self, in_features, hidden_features, out_features, num_layers, dropout=0.5):
        super(GCN, self).__init__()
        self.convs = nn.ModuleList([GraphConvolution(in_features, hidden_features)] +
                                    [GraphConvolution(hidden_features, hidden_features) for _ in range(num_layers-1)])
        self.fc = nn.Linear(hidden_features, out_features)
        self.dropout = dropout

    def forward(self, x, adj):
        # Apply first convolution, ReLU, and dropout
        x = self.convs[0](x, adj)
        x = F.relu(x)
        x = F.dropout(x, training=self.training, p=self.dropout)

        # Apply remaining convolutions
        for conv in self.convs[1:]:
            x = conv(x, adj)

        return F.log_softmax(x, dim=1)


def prepare_data(in_features, num_nodes, num_classes, verbose=False):
    '''Dummy data.'''

    # Generate random node features (num_nodes x out_features)
    features = torch.rand(num_nodes, in_features)

    # Generate a random adjacency matrix (num_nodes x num_nodes)
    # Ensuring it is symmetric and adding self-loops
    adj = torch.rand(num_nodes, num_nodes)
    adj = (adj + adj.T) / 2  # Making it symmetric
    adj += torch.eye(num_nodes)  # Adding self-loops


    # Generate random labels (num_nodes)
    labels = torch.randint(0, num_classes, (num_nodes,))

    # Indices for training and validation nodes
    idx_train = torch.randint(0, num_nodes, (60,))  # 60 random training indices
    idx_val = torch.randint(0, num_nodes, (20,))    # 20 random validation indices

    if verbose: 
        # Print the shapes of the generated tensors
        print("Features shape:", features.shape)  # Should be (num_nodes, out_features)
        print("Adjacency matrix shape:", adj.shape)  # Should be (num_nodes, num_nodes)
        print("Labels shape:", labels.shape)  # Should be (num_nodes,)
        print("Training indices:", idx_train)
        print("Validation indices:", idx_val)

    return features, adj, labels, idx_train, idx_val


# Helper function to compute accuracy
def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).float()
    correct = correct.sum()
    return correct / len(labels)



if __name__ == '__main__':

    # Initialize paramters usage
    num_nodes = 2708  # Cora dataset
    in_features = 1433  # Cora dataset
    hidden_features = 16  # As per the paper
    out_features = 7  # Cora dataset (number of classes)
    num_layers = 2  # As per the paper
    dropout = 0.5  # As per the paper

    epochs=200
    lr=0.01
    weight_decay=5e-4


    device = torch.device('mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu')
    print("Running on: {}".format(device))

    features, adj, labels, idx_train, idx_val = prepare_data(in_features, num_nodes, out_features)


    model = GCN(in_features, hidden_features, out_features, num_layers).to(device)
    optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_val_acc = 0
    best_model = None

    epochs = 100
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()

        # Move entires to the same device
        features = features.to(device)
        adj = adj.to(device)
        labels = labels.to(device)
        idx_train = idx_train.to(device)
        idx_val = idx_val.to(device)

        output = model(features, adj)
        loss_train = F.nll_loss(output[idx_train], labels[idx_train])
        acc_train = accuracy(output[idx_train], labels[idx_train])
        loss_train.backward()
        optimizer.step()

        # Validation
        model.eval()
        output = model(features, adj)
        loss_val = F.nll_loss(output[idx_val], labels[idx_val])
        acc_val = accuracy(output[idx_val], labels[idx_val])
        
        if acc_val > best_val_acc:
            best_val_acc = acc_val
            best_model = model.state_dict()
        
        print(f'Epoch: {epoch+1:03d}, Train Loss: {loss_train.item():.4f}, Train Acc: {acc_train:.4f}, Val Loss: {loss_val.item():.4f}, Val Acc: {acc_val:.4f}')

    print("\nTraining Done.")
