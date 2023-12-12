# Graph Neural Network Models in PyTorch Geometric
## Overview
This repository contains implementations of various GNN models using PyTorch Geometric, including GIN, GCN, GAT, and GraphGPS. The models demonstrate tasks such as node classification and graph classification on datasets Cora, IMDB, ENZYMES, and PascalVOC.

## Setting Up the Conda Environment
### Prerequisites
Ensure you have [Anaconda](https://www.anaconda.com/products/individual) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html) installed.

### Creating the Environment
To create a Conda environment using the `environment.yml` file, follow these steps:

1. Open terminal (or Anaconda Prompt if on Windows).
2. Navigate to the directory containing the `environment.yml` file.
3. Run the following command:
```bash
   conda env create -f environment.yml
```

### Activating the Environment
Once the environment is created, you can activate it using:
```bash
   conda activate DSC180A-Quarter1
```

## Usage
Run each model script as follows:
#### For the GIN model:
```bash
python GIN.py
```
#### For the GCN model:
```bash
python GCN.py
```

#### For the GAT model:
```bash
python GAT.py
```

#### For the GraphGPS model:
```bash
python GraphGPS.py
```