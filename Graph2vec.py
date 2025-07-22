import argparse
import datetime
import torch
from karateclub import Graph2Vec
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from torch_geometric.utils import to_networkx
from dataset.ctp_pyg import CrossChainPairDataset
from utils.result import save_classification_report
from sklearn.model_selection import train_test_split, KFold

def pad_tensor(tensor, target_size, dim=1):
    padding_size = target_size - tensor.size(dim)
    if padding_size > 0:
        padding = torch.zeros(tensor.size(0), padding_size, device=tensor.device)
        tensor = torch.cat((tensor, padding), dim=dim)
    return tensor

def run_experiment(X, labels, n_splits=5, random_state=42):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    fold_results = []

    for fold, (train_idx, test_idx) in enumerate(kf.split(X)):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = [labels[i] for i in train_idx], [labels[i] for i in test_idx]

        model = MLPClassifier()
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        report_dict = classification_report(y_test, y_pred, output_dict=True)
        print(report_dict)
        save_classification_report("Graph2vec", report_dict, "./result/Graph2vec.csv")

    return fold_results

def main(data_path: str, **kwargs):
    dataset = CrossChainPairDataset(
        root=data_path,
    )

    semantics = ['Withdraw','Deposit','Off-chain', 'Not Attack']
    print(datetime.datetime.now(), 'start preprocessing')

    graph_list = []
    label_list = []

    for i, data in enumerate(dataset):
        target_feature_dim = 524
        for node_type in data.node_types:
            data[node_type].x = pad_tensor(data[node_type].x, target_feature_dim)
        for edge_type in data.edge_types:
            data[edge_type].edge_attr = pad_tensor(data[edge_type].edge_attr, target_feature_dim)

        if data.label not in semantics:
            continue

        graph_list.append(to_networkx(data.to_homogeneous())) 
        label_list.append(data.label)

    g2v = Graph2Vec()
    g2v.fit(graph_list)
    X = g2v.get_embedding()

    print(datetime.datetime.now(), 'start experiments')
    experiment_results = run_experiment(X, label_list, n_splits=5)
    print(datetime.datetime.now(), 'end experiments')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, required=True)
    parser.add_argument('--hidden_channels', type=int, default=64)
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--num_heads', type=int, default=4)
    parser.add_argument('--epoch', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--weight_decay', type=float, default=5e-4)
    args = parser.parse_args()
    main(
        data_path=args.data_path,
        model_args=dict(
            hidden_channels=args.hidden_channels,
            num_layers=args.num_layers,
            num_heads=args.num_heads,
        ), **{
            'lr': args.lr,
            'weight_decay': args.weight_decay,
            'epoch': args.epoch,
            'batch_size': args.batch_size,
        }
    )


