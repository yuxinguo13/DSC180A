# %%
import torch
from torch.nn import Linear, ModuleList
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch_geometric.nn import GPSConv, GCNConv, GATConv, LayerNorm, global_mean_pool, GlobalAttention
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
from torch_geometric.datasets import TUDataset
from torch_geometric.datasets import LRGBDataset
from torch_geometric.transforms import NormalizeFeatures
from torch_geometric.data import DataLoader
from torch.utils.data import random_split


# %%
class GPS_cora(torch.nn.Module):
    def __init__(self, in_channels, out_channels, num_layers, new_channels, dropout_rate=0.5):
        super().__init__()
        self.input_transform = Linear(in_channels, new_channels)
        self.convs = torch.nn.ModuleList()
        self.norms = torch.nn.ModuleList()

        self.convs = ModuleList()
        for _ in range(num_layers):
            conv_layer = GCNConv(new_channels, new_channels)
            self.convs.append(GPSConv(new_channels, conv_layer))

        self.dropout = torch.nn.Dropout(dropout_rate)
        self.output = Linear(new_channels, out_channels)

    def forward(self, x, edge_index):
        x = self.input_transform(x)
        for conv, norm in zip(self.convs, self.norms):
            x = conv(x, edge_index)
            x = norm(x)
            x = F.relu(x)
            x = self.dropout(x)
        return F.log_softmax(self.output(x), dim=1)

# %%
Cora_dataset = Planetoid(root='/tmp/Cora', name='Cora')
Cora_data = Cora_dataset[0]

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
new_channels = 1440
Cora_GPSmodel = GPS_cora(Cora_dataset.num_features, Cora_dataset.num_classes, num_layers=10, new_channels=new_channels).to(device)
Cora_data = Cora_data.to(device)
Cora_GPSoptimizer = torch.optim.Adam(Cora_GPSmodel.parameters(), lr=0.005, weight_decay=1e-5)

def train():
    Cora_GPSmodel.train()
    Cora_GPSoptimizer.zero_grad()
    out = Cora_GPSmodel(Cora_data.x, Cora_data.edge_index)
    loss = F.nll_loss(out[Cora_data.train_mask], Cora_data.y[Cora_data.train_mask])
    loss.backward()
    Cora_GPSoptimizer.step()

# Training loop
for epoch in range(500):
    train()

Cora_GPSmodel.eval()
_, Cora_GPSpred = Cora_GPSmodel(Cora_data.x, Cora_data.edge_index).max(dim=1)
Cora_GPSacc = int(Cora_GPSpred[Cora_data.test_mask].eq(Cora_data.y[Cora_data.test_mask]).sum().item()) / int(Cora_data.test_mask.sum()) * 100
Cora_GPSacc_train = int(Cora_GPSpred[Cora_data.train_mask].eq(Cora_data.y[Cora_data.train_mask]).sum().item()) / int(Cora_data.train_mask.sum()) * 100

print('Accuracy of GPS on train Cora:', Cora_GPSacc_train)
print('Accuracy of GPS on test Cora:', Cora_GPSacc)

# %%
class GPS_IMDB(torch.nn.Module):
    def __init__(self, num_features, num_classes, num_layers, new_channels, dropout_rate=0.5):
        super().__init__()
        self.input_transform = Linear(num_features, new_channels)
        self.convs = torch.nn.ModuleList()
        self.norms = torch.nn.ModuleList()
        
        for _ in range(num_layers):
            conv_layer = GCNConv(new_channels, new_channels)
            self.convs.append(GPSConv(new_channels, conv_layer))
            self.norms.append(LayerNorm(new_channels))

        self.dropout = torch.nn.Dropout(dropout_rate)
        self.output = Linear(new_channels, num_classes)
        
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
        batch = data.batch

        for conv, norm in zip(self.convs, self.norms):
            x = conv(x, edge_index)
            x = norm(x)
            x = F.relu(x)
            x = self.dropout(x)

        x = global_mean_pool(x, batch)
        return F.log_softmax(self.output(x), dim=1)

# %%
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
IMDB_dataset = TUDataset(root='/tmp/IMDB-BINARY', name='IMDB-BINARY')
IMDB_GPSmodel = GPS_IMDB(num_features=3, num_classes = 2, num_layers=3, new_channels=3, dropout_rate=0.5).to(device)
IMDB_GPSoptimizer = torch.optim.Adam(IMDB_GPSmodel.parameters(), lr=0.005, weight_decay=5e-4)

IMDB_dataset = TUDataset(root='/tmp/IMDB-BINARY', name='IMDB-BINARY')
IMDB_GPSmodel = GPS_IMDB(num_features=IMDB_dataset.num_node_features, num_classes=IMDB_dataset.num_classes, 
                         num_layers=3, new_channels=3, dropout_rate=0.5).to(device)

IMDB_GPSoptimizer = torch.optim.Adam(IMDB_GPSmodel.parameters(), lr=0.005, weight_decay=5e-4)


total_graphs = len(IMDB_dataset)
train_size = int(total_graphs * 0.8)
test_size = total_graphs - train_size
train_dataset, test_dataset = random_split(IMDB_dataset, [train_size, test_size])
train_loader = DataLoader(train_dataset, batch_size=100, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=100, shuffle=False)

def train():
    IMDB_GPSmodel.train()
    total_loss = 0
    for data in train_loader:
        IMDB_GPSmodel.train()
        IMDB_GPSoptimizer.zero_grad()
        out = IMDB_GPSmodel(data)
        loss = F.nll_loss(out, data.y)
        loss.backward()
        IMDB_GPSoptimizer.step()

for epoch in range(500):
    train()

IMDB_GPSmodel.eval()
correct = 0
for data in train_loader:
    data = data.to(device)
    _, pred = IMDB_GPSmodel(data).max(dim=1)
    correct += pred.eq(data.y).sum().item()

IMDB_GPSacc = correct / len(train_loader)
print('Accuracy of GPS on train IMDB:', IMDB_GPSacc)

IMDB_GPSmodel.eval()
correct = 0
for data in test_loader:
    data = data.to(device)
    _, pred = IMDB_GPSmodel(data).max(dim=1)
    correct += pred.eq(data.y).sum().item()

IMDB_GPSacc_test = correct / len(test_loader)
print('Accuracy of GPS on test IMDB:', IMDB_GPSacc_test)

# %%
class GraphGPS_ENZYMES(torch.nn.Module):
    def __init__(self, num_features, num_classes, num_layers, hidden_dim, heads, dropout_rate=0.5):
        super().__init__()

        # Input linear transformation
        self.linear = nn.Linear(num_features, hidden_dim)

        # Define new_channels based on hidden_dim and heads
        new_channels = hidden_dim
        self.convs = torch.nn.ModuleList()
        self.norms = torch.nn.ModuleList()
        for _ in range(num_layers):
            conv_layer = GATConv(hidden_dim, hidden_dim, heads=heads, dropout=dropout_rate, concat=False)
            self.convs.append(GPSConv(new_channels, conv_layer))
            self.norms.append(LayerNorm(new_channels))
        
        # Output linear layer
        self.output = nn.Linear(new_channels, num_classes)

    def forward(self, x, edge_index, batch):
        x = self.linear(x)
        
        for conv, norm in zip(self.convs, self.norms):
            x = conv(x, edge_index)
            x = norm(x)  # Apply LayerNorm
            x = F.relu(x)

        x = global_mean_pool(x, batch)
        return F.log_softmax(self.output(x), dim=1)

# %%
ENZYMES_dataset = TUDataset(root='/tmp/ENZYMES', name='ENZYMES')
ENZYMES_loader = DataLoader(ENZYMES_dataset, batch_size=32, shuffle=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Instantiate your GraphGPS_ENZYMES model
num_features = ENZYMES_dataset.num_features
num_classes = ENZYMES_dataset.num_classes
model = GraphGPS_ENZYMES(num_features, num_classes, num_layers=3, hidden_dim=128, heads=2, dropout_rate=0.5).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# Training function for GraphGPS_ENZYMES
def train_GPS(model, loader, optimizer):
    model.train()
    for data in loader:
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data.x, data.edge_index, data.batch)
        loss = F.nll_loss(out, data.y)
        loss.backward()
        optimizer.step()

# Training loop
num_epochs = 500
for epoch in range(num_epochs):
    train_GPS(model, ENZYMES_loader, optimizer)

# Evaluation function
def evaluate(model, loader):
    model.eval()
    y_true = []
    y_pred = []
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            out = model(data.x, data.edge_index, data.batch)
            pred = out.max(dim=1)[1]
            y_true.extend(data.y.cpu().numpy())
            y_pred.extend(pred.cpu().numpy())
    accuracy = accuracy_score(y_true, y_pred)
    return accuracy

# Split the dataset into train and test sets
total_graphs = len(ENZYMES_dataset)
train_size = int(total_graphs * 0.8)
test_size = total_graphs - train_size

train_dataset, test_dataset = random_split(ENZYMES_dataset, [train_size, test_size])
train_loader = DataLoader(train_dataset, batch_size=100, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=100, shuffle=False)

# Evaluate the model on the training and test data
train_accuracy = evaluate(model, train_loader)
test_accuracy = evaluate(model, test_loader)
print('Accuracy of GraphGPS on train ENZYMES:', train_accuracy)
print('Accuracy of GraphGPS on test ENZYMES:', test_accuracy)

# %%
class GPS_PascalVOC(torch.nn.Module):
    def __init__(self, in_channels, out_channels, num_layers, new_channels, dropout_rate=0.5):
        super().__init__()
        self.input_transform = Linear(in_channels, new_channels)
        self.convs = torch.nn.ModuleList()
        self.norms = torch.nn.ModuleList()

        for _ in range(num_layers):
            conv_layer = GCNConv(new_channels, new_channels)
            self.convs.append(GPSConv(new_channels, conv_layer))
            self.norms.append(LayerNorm(new_channels))

        self.dropout = torch.nn.Dropout(dropout_rate)
        self.output = Linear(new_channels, num_classes)
        self.dropout = torch.nn.Dropout(dropout_rate)
        self.output = Linear(new_channels, out_channels)

    def forward(self, x, edge_index):
        x = self.input_transform(x)
        for conv, norm in zip(self.convs, self.norms):
            x = conv(x, edge_index)
            x = norm(x)
            x = F.relu(x)
            x = self.dropout(x)
        return F.log_softmax(self.output(x), dim=1)

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
new_channels = 12
PascalVOC_GPSmodel = GPS_PascalVOC(PascalVOC_dataset.num_features, PascalVOC_dataset.num_classes, num_layers=10, new_channels=new_channels).to(device)
PascalVOC_data = PascalVOC_data.to(device)
PascalVOC_GPSoptimizer = torch.optim.Adam(PascalVOC_GPSmodel.parameters(), lr=0.001, weight_decay=1e-5)

def train():
    PascalVOC_GPSmodel.train()
    PascalVOC_GPSoptimizer.zero_grad()
    out = PascalVOC_GPSmodel(PascalVOC_data.x, PascalVOC_data.edge_index)
    loss = F.nll_loss(out[PascalVOC_data.train_mask], PascalVOC_data.y[PascalVOC_data.train_mask])
    loss.backward()
    PascalVOC_GPSoptimizer.step()

for epoch in range(100):
    train()

PascalVOC_GPSmodel.eval()
_, PascalVOC_GPSpred = PascalVOC_GPSmodel(PascalVOC_data.x, PascalVOC_data.edge_index).max(dim=1)
PascalVOC_GPSacc = int(PascalVOC_GPSpred[PascalVOC_data.test_mask].eq(PascalVOC_data.y[PascalVOC_data.test_mask]).sum().item()) / int(PascalVOC_data.test_mask.sum()) * 100
PascalVOC_GPSacc_train = int(PascalVOC_GPSpred[PascalVOC_data.train_mask].eq(PascalVOC_data.y[PascalVOC_data.train_mask]).sum().item()) / int(PascalVOC_data.train_mask.sum()) * 100
print('Accuracy of GPS on train PascalVOC:', PascalVOC_GPSacc_train)
print('Accuracy of GPS on test PascalVOC:', PascalVOC_GPSacc)


# %%



