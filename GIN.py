# %%
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.datasets import Planetoid
from torch_geometric.datasets import TUDataset
from torch_geometric.datasets import LRGBDataset
from torch.utils.data import random_split
from torch_geometric.nn import GINConv, global_mean_pool
from torch_geometric.nn.norm import BatchNorm

# %%
class GINNet_Cora(torch.nn.Module):
    def __init__(self, num_features, num_classes):
        super(GINNet_Cora, self).__init__()
        self.conv1 = GINConv(torch.nn.Sequential(
            torch.nn.Linear(num_features, 16),
            torch.nn.ReLU(),
            torch.nn.Linear(16, 16),
            torch.nn.ReLU()
        ))
        self.conv2 = GINConv(torch.nn.Sequential(
            torch.nn.Linear(16, 16),
            torch.nn.ReLU(),
            torch.nn.Linear(16, num_classes)
        ))

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)

        return F.log_softmax(x, dim=1)

# %%
Cora_dataset = Planetoid(root='/tmp/Cora', name='Cora')
Cora_data = Cora_dataset[0]

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
Cora_GINmodel = GINNet_Cora(Cora_dataset.num_features, Cora_dataset.num_classes).to(device)
Cora_data = Cora_data.to(device)
Cora_GINoptimizer = torch.optim.Adam(Cora_GINmodel.parameters(), lr=0.005, weight_decay=5e-4)

def train():
    out = Cora_GINmodel(Cora_data)
    Cora_GINoptimizer.zero_grad()
    loss = F.nll_loss(out[Cora_data.train_mask], Cora_data.y[Cora_data.train_mask])
    loss.backward()
    Cora_GINoptimizer.step()

for epoch in range(50):
    train()

Cora_GINmodel.eval()
_, Cora_GINpred = Cora_GINmodel(Cora_data).max(dim=1)
Cora_GINacc = int(Cora_GINpred[Cora_data.test_mask].eq(Cora_data.y[Cora_data.test_mask]).sum().item()) / int(Cora_data.test_mask.sum()) * 100
Cora_GINacc_train = int(Cora_GINpred[Cora_data.train_mask].eq(Cora_data.y[Cora_data.train_mask]).sum().item()) / int(Cora_data.train_mask.sum()) * 100
print('Accuracy of GIN on train Cora:', Cora_GINacc_train)
print('Accuracy of GIN on test Cora:', Cora_GINacc)

# %%
class GINGraphNet_IMDB(torch.nn.Module):
    def __init__(self, num_features, num_classes):
        super(GINGraphNet_IMDB, self).__init__()
        self.conv1 = GINConv(torch.nn.Sequential(
            torch.nn.Linear(num_features, 16),
            torch.nn.ReLU(),
            torch.nn.Linear(16, 16),
            torch.nn.ReLU()
        ))
        self.conv2 = GINConv(torch.nn.Sequential(
            torch.nn.Linear(16, 16),
            torch.nn.ReLU(),
            torch.nn.Linear(16, num_classes)
        ))
        
    
    def forward(self, data):
        def degree_feature(data):
            deg = torch.zeros(data.num_nodes, dtype=torch.float).to(data.edge_index.device)
            # Counting source and target edges
            deg.index_add_(0, data.edge_index[0], torch.ones(data.edge_index.size(1), device=data.edge_index.device))
            deg.index_add_(0, data.edge_index[1], torch.ones(data.edge_index.size(1), device=data.edge_index.device))
            return deg.view(-1, 1)
        
        const_feat = torch.ones((data.num_nodes, 1))
        deg_feat = degree_feature(data)
        rand_feat = torch.rand((data.num_nodes, 1))

        x = torch.cat([const_feat, deg_feat, rand_feat], dim=1)
        edge_index = data.edge_index

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = global_mean_pool(x, data.batch)

        return F.log_softmax(x, dim=1)

# %%
from torch_geometric.data import DataLoader

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
IMDB_dataset = TUDataset(root='/tmp/IMDB-BINARY', name='IMDB-BINARY')
IMDB_GINmodel = GINGraphNet_IMDB(3, IMDB_dataset.num_classes).to(device)
IMDB_GINoptimizer = torch.optim.Adam(IMDB_GINmodel.parameters(), lr=0.005, weight_decay=5e-4)

total_graphs = len(IMDB_dataset)
train_size = int(total_graphs * 0.9)
test_size = total_graphs - train_size

train_dataset, test_dataset = random_split(IMDB_dataset, [train_size, test_size])
train_loader = DataLoader(train_dataset, batch_size=100, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=100, shuffle=False)

def train_GIN():
    IMDB_GINmodel.train()
    for data in train_loader:
        data = data.to(device)
        IMDB_GINoptimizer.zero_grad()
        out = IMDB_GINmodel(data)
        loss = F.nll_loss(out, data.y)
        loss.backward()
        IMDB_GINoptimizer.step()

for epoch in range(2000):
    train_GIN()

IMDB_GINmodel.eval()
correct = 0
for data in train_loader:
    data = data.to(device)
    _, pred = IMDB_GINmodel(data).max(dim=1)
    correct += pred.eq(data.y).sum().item()

IMDB_GINacc = correct / len(train_loader)
print('Accuracy of GIN on train IMDB:', IMDB_GINacc)

IMDB_GINmodel.eval()
correct = 0
for data in test_loader:
    data = data.to(device)
    _, pred = IMDB_GINmodel(data).max(dim=1)
    correct += pred.eq(data.y).sum().item()

IMDB_GINacc_test = correct / len(test_loader)
print('Accuracy of GIN on test IMDB:', IMDB_GINacc_test)

# %%
class GINGraphNet_ENZYMES(torch.nn.Module):
    def __init__(self, num_features, num_classes):
        super(GINGraphNet_ENZYMES, self).__init__()
        nn1 = torch.nn.Sequential(torch.nn.Linear(num_features, 16), torch.nn.ReLU(), torch.nn.Linear(16, 16))
        self.conv1 = GINConv(nn1)
        nn2 = torch.nn.Sequential(torch.nn.Linear(16, 8), torch.nn.ReLU(), torch.nn.Linear(8, 8))
        self.conv2 = GINConv(nn2)
        self.fc = torch.nn.Linear(8, num_classes)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = global_mean_pool(x, data.batch)
        return F.log_softmax(self.fc(x), dim=1)

# %%
from torch_geometric.data import DataLoader
ENZYMES_dataset = TUDataset(root='/tmp/ENZYMES', name='ENZYMES')
ENZYMES_loader = DataLoader(ENZYMES_dataset, batch_size=32, shuffle=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

ENZYMES_GINmodel = GINGraphNet_ENZYMES(ENZYMES_dataset.num_features, ENZYMES_dataset.num_classes).to(device)
ENZYMES_GINoptimizer = torch.optim.Adam(ENZYMES_GINmodel.parameters(), lr=0.01)

total_graphs = len(ENZYMES_dataset)
train_size = int(total_graphs * 0.8)
test_size = total_graphs - train_size

train_dataset, test_dataset = random_split(ENZYMES_dataset, [train_size, test_size])
train_loader = DataLoader(train_dataset, batch_size=200, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=200, shuffle=False)

def train_GIN():
    ENZYMES_GINmodel.train()
    for data in train_loader:
        data = data.to(device)
        ENZYMES_GINoptimizer.zero_grad()
        out = ENZYMES_GINmodel(data)
        loss = F.nll_loss(out, data.y)
        loss.backward()
        ENZYMES_GINoptimizer.step()

for epoch in range(5000):
    train_GIN()

ENZYMES_GINmodel.eval()
correct = 0
for data in train_loader:
    data = data.to(device)
    _, pred = ENZYMES_GINmodel(data).max(dim=1)
    correct += pred.eq(data.y).sum().item()

ENZYMES_GINacc = correct / len(train_loader)
print('Accuracy of GIN on train ENZYMES:', ENZYMES_GINacc)


ENZYMES_GINmodel.eval()
correct = 0
for data in test_loader:
    data = data.to(device)
    _, pred = ENZYMES_GINmodel(data).max(dim=1)
    correct += pred.eq(data.y).sum().item()

ENZYMES_GINacc_test = correct / len(test_loader)
print('Accuracy of GIN on test ENZYMES:', ENZYMES_GINacc_test)

# %%
class GINNet_PascalVOC(torch.nn.Module):
    def __init__(self, num_features, num_classes):
        super(GINNet_PascalVOC, self).__init__()
        self.conv1 = GINConv(torch.nn.Sequential(
            torch.nn.Linear(num_features, 16),
            torch.nn.ReLU(),
            torch.nn.Linear(16, 16),
            torch.nn.ReLU()
        ))
        self.conv2 = GINConv(torch.nn.Sequential(
            torch.nn.Linear(16, 16),
            torch.nn.ReLU(),
            torch.nn.Linear(16, num_classes)
        ))

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)

        return F.log_softmax(x, dim=1)

# %%
from torch_geometric.data import DataLoader
PascalVOC_dataset = LRGBDataset(root='/tmp/PascalVOC-SP', name='PascalVOC-SP')
PascalVOC_data = PascalVOC_dataset[0]

def create_masks(num_nodes, train_percent=0.8):
    indices = torch.randperm(num_nodes)
    train_size = int(num_nodes * train_percent)
    
    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(num_nodes, dtype=torch.bool)
    
    train_mask[indices[:train_size]] = True
    test_mask[indices[train_size:]] = True
    
    return train_mask, test_mask

train_mask, test_mask = create_masks(num_nodes=460, train_percent=0.8)
PascalVOC_data.train_mask = train_mask
PascalVOC_data.test_mask = test_mask

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
PascalVOC_GINmodel = GINNet_PascalVOC(PascalVOC_dataset.num_features, PascalVOC_dataset.num_classes).to(device)
PascalVOC_data = PascalVOC_data.to(device)
PascalVOC_GINoptimizer = torch.optim.Adam(PascalVOC_GINmodel.parameters(), lr=0.005, weight_decay=5e-4)

def train():
    out = PascalVOC_GINmodel(PascalVOC_data)
    PascalVOC_GINoptimizer.zero_grad()
    loss = F.nll_loss(out[PascalVOC_data.train_mask], PascalVOC_data.y[PascalVOC_data.train_mask])
    loss.backward()
    PascalVOC_GINoptimizer.step()

for epoch in range(50):
    train()

PascalVOC_GINmodel.eval()
_, PascalVOC_GINpred = PascalVOC_GINmodel(PascalVOC_data).max(dim=1)
PascalVOC_GINacc = int(PascalVOC_GINpred[PascalVOC_data.test_mask].eq(PascalVOC_data.y[PascalVOC_data.test_mask]).sum().item()) / int(PascalVOC_data.test_mask.sum()) * 100
PascalVOC_GINacc_train = int(PascalVOC_GINpred[PascalVOC_data.train_mask].eq(PascalVOC_data.y[PascalVOC_data.train_mask]).sum().item()) / int(PascalVOC_data.train_mask.sum()) * 100
print('Accuracy of GIN on train PascalVOC:', PascalVOC_GINacc_train)
print('Accuracy of GIN on test PascalVOC:', PascalVOC_GINacc)


# %%
