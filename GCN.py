# %%
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.datasets import Planetoid
from torch_geometric.datasets import TUDataset
from torch_geometric.datasets import LRGBDataset
from torch.utils.data import random_split
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.nn.norm import BatchNorm

# %%
class GCNNet_Cora(torch.nn.Module):
    def __init__(self, num_features, num_classes):
        super(GCNNet_Cora, self).__init__()
        self.conv1 = GCNConv(num_features, 128)
        self.conv2 = GCNConv(128, 16)
        self.fc = torch.nn.Linear(16, num_classes)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)

        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)

        x = self.fc(x)
    
        return F.log_softmax(x, dim=1)

# %%
Cora_dataset = Planetoid(root='/tmp/Cora', name='Cora')
Cora_data = Cora_dataset[0]

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
Cora_GCNmodel = GCNNet_Cora(Cora_dataset.num_features, Cora_dataset.num_classes).to(device)
Cora_data = Cora_data.to(device)
Cora_GCNoptimizer = torch.optim.Adam(Cora_GCNmodel.parameters(), lr=0.005, weight_decay=5e-4)

def train():
    out = Cora_GCNmodel(Cora_data)
    Cora_GCNoptimizer.zero_grad()
    loss = F.nll_loss(out[Cora_data.train_mask], Cora_data.y[Cora_data.train_mask])
    loss.backward()
    Cora_GCNoptimizer.step()

for epoch in range(50):
    train()

Cora_GCNmodel.eval()
_, Cora_GCNpred = Cora_GCNmodel(Cora_data).max(dim=1)
Cora_GCNacc = int(Cora_GCNpred[Cora_data.test_mask].eq(Cora_data.y[Cora_data.test_mask]).sum().item()) / int(Cora_data.test_mask.sum()) * 100
Cora_GCNacc_train = int(Cora_GCNpred[Cora_data.train_mask].eq(Cora_data.y[Cora_data.train_mask]).sum().item()) / int(Cora_data.train_mask.sum()) * 100
print('Accuracy of GCN on train Cora:', Cora_GCNacc_train)
print('Accuracy of GCN on test Cora:', Cora_GCNacc)

# %%
class GCNGraphNet_IMDB(torch.nn.Module):
    def __init__(self, num_features, num_classes):
        super(GCNGraphNet_IMDB, self).__init__()
        self.conv1 = GCNConv(num_features, 16)
        self.conv2 = GCNConv(16, 16)
        self.fc = torch.nn.Linear(16, num_classes)

    def forward(self, data):
        def degree_feature(data):
            deg = torch.zeros(data.num_nodes, dtype=torch.float).to(data.edge_index.device)
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
IMDB_GCNmodel = GCNGraphNet_IMDB(3, IMDB_dataset.num_classes).to(device)
IMDB_GCNoptimizer = torch.optim.Adam(IMDB_GCNmodel.parameters(), lr=0.005, weight_decay=5e-4)

total_graphs = len(IMDB_dataset)
train_size = int(total_graphs * 0.9)
test_size = total_graphs - train_size

train_dataset, test_dataset = random_split(IMDB_dataset, [train_size, test_size])
train_loader = DataLoader(train_dataset, batch_size=100, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=100, shuffle=False)

def train_GCN():
    IMDB_GCNmodel.train()
    for data in train_loader:
        data = data.to(device)
        IMDB_GCNoptimizer.zero_grad()
        out = IMDB_GCNmodel(data)
        loss = F.nll_loss(out, data.y)
        loss.backward()
        IMDB_GCNoptimizer.step()

for epoch in range(200):
    train_GCN()

IMDB_GCNmodel.eval()
correct = 0
for data in train_loader:
    data = data.to(device)
    _, pred = IMDB_GCNmodel(data).max(dim=1)
    correct += pred.eq(data.y).sum().item()

IMDB_GCNacc = correct / len(train_loader)
print('Accuracy of GCN on train IMDB:', IMDB_GCNacc)

IMDB_GCNmodel.eval()
correct = 0
for data in test_loader:
    data = data.to(device)
    _, pred = IMDB_GCNmodel(data).max(dim=1)
    correct += pred.eq(data.y).sum().item()

IMDB_GCNacc_test = correct / len(test_loader)
print('Accuracy of GCN on test IMDB:', IMDB_GCNacc_test)

# %%
class GCNGraphNet_Enzymes(torch.nn.Module):
    def __init__(self, num_features, num_classes, hidden_dim=16, num_layers=2, dropout=0.5):
        super(GCNGraphNet_Enzymes, self).__init__()
        
        self.num_layers = num_layers
        self.convs = torch.nn.ModuleList([GCNConv(num_features if i == 0 else hidden_dim, hidden_dim) for i in range(num_layers)])
        self.convs.append(GCNConv(hidden_dim, num_classes))
        self.bns = torch.nn.ModuleList([torch.nn.BatchNorm1d(hidden_dim) for _ in range(num_layers)])
        self.dropout = dropout

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        for i in range(self.num_layers):
            x = F.relu(self.bns[i](self.convs[i](x, edge_index)))
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, edge_index)
        x = global_mean_pool(x, data.batch)
        return F.log_softmax(x, dim=1)


# %%
from torch_geometric.data import DataLoader
ENZYMES_dataset = TUDataset(root='/tmp/ENZYMES', name='ENZYMES')
ENZYMES_loader = DataLoader(ENZYMES_dataset, batch_size=32, shuffle=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

ENZYMES_GCNmodel = GCNGraphNet_Enzymes(ENZYMES_dataset.num_features, ENZYMES_dataset.num_classes).to(device)
ENZYMES_GCNoptimizer = torch.optim.Adam(ENZYMES_GCNmodel.parameters(), lr=0.005)

total_graphs = len(ENZYMES_dataset)
train_size = int(total_graphs * 0.9)
test_size = total_graphs - train_size

train_dataset, test_dataset = random_split(ENZYMES_dataset, [train_size, test_size])
train_loader = DataLoader(train_dataset, batch_size=100, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=100, shuffle=False)

def train_GCN():
    ENZYMES_GCNmodel.train()
    for data in train_loader:
        data = data.to(device)
        ENZYMES_GCNoptimizer.zero_grad()
        out = ENZYMES_GCNmodel(data)
        loss = F.nll_loss(out, data.y)
        loss.backward()
        ENZYMES_GCNoptimizer.step()

for epoch in range(200):
    train_GCN()


ENZYMES_GCNmodel.eval()
correct = 0
for data in train_loader:
    data = data.to(device)
    _, pred = ENZYMES_GCNmodel(data).max(dim=1)
    correct += pred.eq(data.y).sum().item()

ENZYMES_GCNacc = correct / len(train_loader)
print('Accuracy of GCN on train ENZYMES:', ENZYMES_GCNacc)


ENZYMES_GCNmodel.eval()
correct = 0
for data in test_loader:
    data = data.to(device)
    _, pred = ENZYMES_GCNmodel(data).max(dim=1)
    correct += pred.eq(data.y).sum().item()

ENZYMES_GCNacc_test = correct / len(test_loader)
print('Accuracy of GCN on test ENZYMES:', ENZYMES_GCNacc_test)

# %%
class GCNGraphNet_PascalVOC(torch.nn.Module):
    def __init__(self, num_features, num_classes):
        super(GCNGraphNet_PascalVOC, self).__init__()
        self.conv1 = GCNConv(num_features, 16)
        self.conv2 = GCNConv(16, num_classes)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
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
PascalVOC_GCNmodel = GCNGraphNet_PascalVOC(PascalVOC_dataset.num_features, PascalVOC_dataset.num_classes).to(device)
PascalVOC_data = PascalVOC_data.to(device)
PascalVOC_GCNoptimizer = torch.optim.Adam(PascalVOC_GCNmodel.parameters(), lr=0.005, weight_decay=5e-4)

def train():
    out = PascalVOC_GCNmodel(PascalVOC_data)
    PascalVOC_GCNoptimizer.zero_grad()
    loss = F.nll_loss(out[PascalVOC_data.train_mask], PascalVOC_data.y[PascalVOC_data.train_mask])
    loss.backward()
    PascalVOC_GCNoptimizer.step()

for epoch in range(50):
    train()

PascalVOC_GCNmodel.eval()
_, PascalVOC_GCNpred = PascalVOC_GCNmodel(PascalVOC_data).max(dim=1)
PascalVOC_GCNacc = int(PascalVOC_GCNpred[PascalVOC_data.test_mask].eq(PascalVOC_data.y[PascalVOC_data.test_mask]).sum().item()) / int(PascalVOC_data.test_mask.sum()) * 100
PascalVOC_GCNacc_train = int(PascalVOC_GCNpred[PascalVOC_data.train_mask].eq(PascalVOC_data.y[PascalVOC_data.train_mask]).sum().item()) / int(PascalVOC_data.train_mask.sum()) * 100
print('Accuracy of GCN on train PascalVOC:', PascalVOC_GCNacc_train)
print('Accuracy of GCN on test PascalVOC:', PascalVOC_GCNacc)


# %%


# %%


# %%


# %%



