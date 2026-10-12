# BridgeShield

BridgeShield is a graph learning framework for detecting cross-chain bridge attacks through risk-aware modeling of cross-chain execution semantics.

Related paper: BridgeShield: Risk-Aware Graph Modeling for Cross-Chain Bridge Attack Detection

## 1. Code Structure

```text
├── BridgeShield.py       # Entry point for model training and evaluation
├── dataset/              # Transaction graph construction and PyG data conversion
├── models/               # BridgeShield and baseline model definitions
├── GAT.py / HEAT.py / HGT.py / RGCN.py / Graph2vec.py    # Baseline
└── requirements.txt      # Dependencies
```

## 2. Environment Setup

The dependencies required for the experiments are listed in `requirements.txt`. Install them using:

```bash
pip install -r requirements.txt
```

## 3. Data Processing

Data processing consists of two steps: transaction pair construction and format conversion.

1. Construct transaction pairs from the raw data of matched transactions on the source and destination chains.

```bash
python ctp.py
```

2. Convert the transaction pairs into PyG format.

```bash
python ctp_pyg.py
```

## 4. Running the Models

After completing data processing, run BridgeShield for training and evaluation:

```bash
python BridgeShield.py
```

Implementations of the baseline methods are also provided:

```bash
python GAT.py
python HEAT.py
python HGT.py
python RGCN.py
python Graph2vec.py
```

