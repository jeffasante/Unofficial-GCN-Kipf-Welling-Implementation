# Unofficial GCN Implementation - Kipf & Welling (2016)

This is an unofficial PyTorch implementation of the Graph Convolutional Network (GCN) from the paper ["Semi-Supervised Classification with Graph Convolutional Networks"](https://arxiv.org/abs/1609.02907) by Thomas Kipf and Max Welling. This implementation leverages the `einops` library for efficient and flexible tensor operations.

## Features
- Graph Convolution using adjacency normalization and self-loops.
- Efficient tensor reshaping and summing using `einops`.
- Dummy data generation for testing the GCN model.

## Install
To install dependencies, run:

```bash
$ pip install torch einops
```

## Usage

Here's how you can test the GCN model using dummy data:

```python
import torch
from gnn import GCN, prepare_data, accuracy

# Initialize parameters
num_nodes = 2708  # Cora dataset
in_features = 1433  # Cora dataset
hidden_features = 16  # As per the paper
out_features = 7  # Cora dataset (number of classes)
num_layers = 2  # As per the paper

# Generate dummy data
features, adj, labels, idx_train, idx_val = prepare_data(in_features, num_nodes, out_features)

# Initialize model
model = GCN(in_features, hidden_features, out_features, num_layers)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

# Training loop
model.train()
output = model(features, adj)
loss_train = torch.nn.functional.nll_loss(output[idx_train], labels[idx_train])
acc_train = accuracy(output[idx_train], labels[idx_train])
print(f"Train Accuracy: {acc_train:.4f}")
```

This implementation supports multi-layer GCNs and can be adapted for real-world graph data beyond dummy inputs.

## License

This repository is open-sourced under the MIT License.