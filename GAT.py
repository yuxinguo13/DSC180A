# %%
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.datasets import Planetoid
from torch_geometric.datasets import TUDataset
from torch_geometric.datasets import LRGBDataset
from torch.utils.data import random_split
from torch_geometric.nn import GATv2Conv, global_mean_pool
from torch_geometric.nn.norm import BatchNorm

# %%
class GATv2Net_Cora(torch.nn.Module): # for Node classification
    def __init__(self, num_features, num_classes):
        super(GATv2Net_Cora, self).__init__()
        self.conv1 = GATv2Conv(num_features, 16, heads=2)
        self.conv2 = GATv2Conv(32, 16, heads=2)
        self.fc = torch.nn.Linear(32, num_classes)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = F.elu(self.conv1(x, edge_index))
        x = F.elu(self.conv2(x, edge_index))
        x = self.fc(x)

        return F.log_softmax(x, dim=1)

# %%
Cora_dataset = Planetoid(root='/tmp/Cora', name='Cora')
Cora_data = Cora_dataset[0]

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
Cora_GATmodel = GATv2Net_Cora(Cora_dataset.num_features, Cora_dataset.num_classes).to(device)
Cora_data = Cora_data.to(device)
Cora_GAToptimizer = torch.optim.Adam(Cora_GATmodel.parameters(), lr=0.005, weight_decay=5e-4)

def train():
    out = Cora_GATmodel(Cora_data)
    Cora_GAToptimizer.zero_grad()
    loss = F.nll_loss(out[Cora_data.train_mask], Cora_data.y[Cora_data.train_mask])
    loss.backward()
    Cora_GAToptimizer.step()

for epoch in range(50):
    train()

Cora_GATmodel.eval()
_, Cora_GATpred = Cora_GATmodel(Cora_data).max(dim=1)
Cora_GATacc = int(Cora_GATpred[Cora_data.test_mask].eq(Cora_data.y[Cora_data.test_mask]).sum().item()) / int(Cora_data.test_mask.sum()) * 100
Cora_GATacc_train = int(Cora_GATpred[Cora_data.train_mask].eq(Cora_data.y[Cora_data.train_mask]).sum().item()) / int(Cora_data.train_mask.sum()) * 100
print('Accuracy of GAT on train Cora:', Cora_GATacc_train)
print('Accuracy of GAT on test Cora:', Cora_GATacc)

# %%
class GATv2GraphNet_IMDB(torch.nn.Module):
    def __init__(self, num_features, num_classes):
        super(GATv2GraphNet_IMDB, self).__init__()
        self.conv1 = GATv2Conv(num_features, 16, heads=2)
        self.conv2 = GATv2Conv(32, 16, heads=2)
        self.fc = torch.nn.Linear(32, num_classes)

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

        x = F.elu(self.conv1(x, edge_index))
        x = F.elu(self.conv2(x, edge_index))
        x = global_mean_pool(x, data.batch)
        x = self.fc(x)

        return F.log_softmax(x, dim=1)

# %%
from torch_geometric.data import DataLoader

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
IMDB_dataset = TUDataset(root='/tmp/IMDB-BINARY', name='IMDB-BINARY')
IMDB_GATmodel = GATv2GraphNet_IMDB(3, IMDB_dataset.num_classes).to(device)
IMDB_GAToptimizer = torch.optim.Adam(IMDB_GATmodel.parameters(), lr=0.005, weight_decay=5e-4)

total_graphs = len(IMDB_dataset)
train_size = int(total_graphs * 0.9)
test_size = total_graphs - train_size

train_dataset, test_dataset = random_split(IMDB_dataset, [train_size, test_size])
train_loader = DataLoader(train_dataset, batch_size=100, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=100, shuffle=False)

def train_GAT():
    IMDB_GATmodel.train()
    for data in train_loader:
        data = data.to(device)
        IMDB_GAToptimizer.zero_grad()
        out = IMDB_GATmodel(data)
        loss = F.nll_loss(out, data.y)
        loss.backward()
        IMDB_GAToptimizer.step()

for epoch in range(200):
    train_GAT()

IMDB_GATmodel.eval()
correct = 0
for data in train_loader:
    data = data.to(device)
    _, pred = IMDB_GATmodel(data).max(dim=1)
    correct += pred.eq(data.y).sum().item()

IMDB_GATacc = correct / len(train_loader)
print('Accuracy of GAT on train IMDB:', IMDB_GATacc)

IMDB_GATmodel.eval()
correct = 0
for data in test_loader:
    data = data.to(device)
    _, pred = IMDB_GATmodel(data).max(dim=1)
    correct += pred.eq(data.y).sum().item()

IMDB_GATacc_test = correct / len(test_loader)
print('Accuracy of GAT on test IMDB:', IMDB_GATacc_test)

# %%
class GATv2GraphNet_ENZYMES(torch.nn.Module):
    def __init__(self, num_features, num_classes):
        super(GATv2GraphNet_ENZYMES, self).__init__()
        self.conv1 = GATv2Conv(num_features, 32, heads=2, dropout=0.2)
        self.conv2 = GATv2Conv(64, 16, heads=2, dropout=0.2)
        self.fc = torch.nn.Linear(32, num_classes)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = self.conv1(x, edge_index)
        x = F.elu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        x = F.elu(x)
        x = F.dropout(x, training=self.training)
        x = global_mean_pool(x, data.batch)

        return F.log_softmax(self.fc(x), dim=1)

# %%
from torch_geometric.data import DataLoader
ENZYMES_dataset = TUDataset(root='/tmp/ENZYMES', name='ENZYMES')
ENZYMES_loader = DataLoader(ENZYMES_dataset, batch_size=32, shuffle=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

ENZYMES_GATmodel = GATv2GraphNet_ENZYMES(ENZYMES_dataset.num_features, ENZYMES_dataset.num_classes).to(device)
ENZYMES_GAToptimizer = torch.optim.Adam(ENZYMES_GATmodel.parameters(), lr=0.01)

total_graphs = len(ENZYMES_dataset)
train_size = int(total_graphs * 0.8)
test_size = total_graphs - train_size

train_dataset, test_dataset = random_split(ENZYMES_dataset, [train_size, test_size])
train_loader = DataLoader(train_dataset, batch_size=200, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=200, shuffle=False)

def train_GCN():
    ENZYMES_GATmodel.train()
    for data in train_loader:
        data = data.to(device)
        ENZYMES_GAToptimizer.zero_grad()
        out = ENZYMES_GATmodel(data)
        loss = F.nll_loss(out, data.y)
        loss.backward()
        ENZYMES_GAToptimizer.step()

for epoch in range(5000):
    train_GCN()

ENZYMES_GATmodel.eval()
correct = 0
for data in train_loader:
    data = data.to(device)
    _, pred = ENZYMES_GATmodel(data).max(dim=1)
    correct += pred.eq(data.y).sum().item()

ENZYMES_GATacc = correct / len(train_loader)
print('Accuracy of GAT on train ENZYMES:', ENZYMES_GATacc)


ENZYMES_GATmodel.eval()
correct = 0
for data in test_loader:
    data = data.to(device)
    _, pred = ENZYMES_GATmodel(data).max(dim=1)
    correct += pred.eq(data.y).sum().item()

ENZYMES_GATacc_test = correct / len(test_loader)
print('Accuracy of GAT on test ENZYMES:', ENZYMES_GATacc_test)

# %%
class GATv2Net_PascalVOC(torch.nn.Module): # for Node classification
    def __init__(self, num_features, num_classes):
        super(GATv2Net_PascalVOC, self).__init__()
        self.conv1 = GATv2Conv(num_features, 16, heads=2)
        self.conv2 = GATv2Conv(32, 16, heads=2)
        self.fc = torch.nn.Linear(32, num_classes)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = F.elu(self.conv1(x, edge_index))
        x = F.elu(self.conv2(x, edge_index))
        x = self.fc(x)

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
PascalVOC_GATmodel = GATv2Net_PascalVOC(PascalVOC_dataset.num_features, PascalVOC_dataset.num_classes).to(device)
PascalVOC_data = PascalVOC_data.to(device)
PascalVOC_GAToptimizer = torch.optim.Adam(PascalVOC_GATmodel.parameters(), lr=0.005, weight_decay=5e-4)

def train():
    out = PascalVOC_GATmodel(PascalVOC_data)
    PascalVOC_GAToptimizer.zero_grad()
    loss = F.nll_loss(out[PascalVOC_data.train_mask], PascalVOC_data.y[PascalVOC_data.train_mask])
    loss.backward()
    PascalVOC_GAToptimizer.step()

for epoch in range(50):
    train()

PascalVOC_GATmodel.eval()
_, PascalVOC_GATpred = PascalVOC_GATmodel(PascalVOC_data).max(dim=1)
PascalVOC_GATacc = int(PascalVOC_GATpred[PascalVOC_data.test_mask].eq(PascalVOC_data.y[PascalVOC_data.test_mask]).sum().item()) / int(PascalVOC_data.test_mask.sum()) * 100
PascalVOC_GATacc_train = int(PascalVOC_GATpred[PascalVOC_data.train_mask].eq(PascalVOC_data.y[PascalVOC_data.train_mask]).sum().item()) / int(PascalVOC_data.train_mask.sum()) * 100
print('Accuracy of GAT on train PascalVOC:', PascalVOC_GATacc_train)
print('Accuracy of GAT on test PascalVOC:', PascalVOC_GATacc)


# %%


# %%



