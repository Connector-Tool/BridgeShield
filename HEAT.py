import argparse
import datetime
import os
from sklearn.model_selection import KFold
os.environ['CUDA_LAUNCH_BLOCKING'] = '0'
from torch_geometric.loader import DataLoader
from dataset.ctp_pyg import CrossChainPairDataset
from models.HEAT import HEAT
from sklearn.metrics import classification_report
from utils.result import save_classification_report
import torch

label_mapping = {
    'Not Attack':0,
    'Deposit':1,
    'Withdraw':2,
    'Off-chain':3
}


def train(data_path: str, model_args: dict, **kwargs):
    print(model_args)
    print(kwargs)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    device = 'cpu' if kwargs.get('gpu', False) == 'False' else device

    dataset = CrossChainPairDataset(root=data_path)
    torch.manual_seed(42)
    labels = torch.tensor([
        label_mapping[data.label] if data.label in label_mapping.keys() else 0
        for data in dataset
    ])
    unique_labels, counts = torch.unique(labels, return_counts=True)
    class_weights = 1. / counts.float()
    weights = class_weights / class_weights.sum()
    weights = weights.to(device)

    k_folds = kwargs.get('k_folds', 5)
    kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)

    for fold, (train_idx, valid_idx) in enumerate(kf.split(dataset)):
        model = HEAT(**{
            "hidden_channels": kwargs.get('hidden_channels', 32),
            "out_channels": 4,
            "metadata": dataset.metadata,
            "num_layers": kwargs.get('num_layers', 4),
            **model_args,
        }).to(device)
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=kwargs.get('lr', 0.001),
            weight_decay=kwargs.get('weight_decay', 5e-4),
        )

        criterion = torch.nn.CrossEntropyLoss(weight = weights)
        model.train()

        train_subset = torch.utils.data.Subset(dataset, train_idx)
        valid_subset = torch.utils.data.Subset(dataset, valid_idx)
        train_loader = DataLoader(train_subset, batch_size=kwargs.get('batch_size', 64), shuffle=True, num_workers=4)
        valid_loader = DataLoader(valid_subset, batch_size=kwargs.get('batch_size', 64), shuffle=False, num_workers=4)

        print("start train")
        for data in train_loader:
            data = data.to(device)
            out = model(data)
            data.label = torch.tensor([label_mapping[label] if label in label_mapping else 0 for label in data.label],
                                      dtype=torch.long).to(device)
            loss = criterion(out, data.label)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            print('{}, loss {}'.format(
                datetime.datetime.now(), loss
            ))


        model.eval()
        val_loss = 0

        print("start test")
        with torch.no_grad():
            all_labels = []
            all_predictions = []

            for data in valid_loader:
                data = data.to(device)
                out = model(data)
                data.label = torch.tensor([label_mapping[label] if label in label_mapping else 0 for label in data.label],
                                          dtype=torch.long).to(device)
                loss = criterion(out, data.label)
                val_loss += loss.item()
                _, predicted = torch.max(out, 1)
                all_labels.extend(data.label.cpu().numpy())
                all_predictions.extend(predicted.cpu().numpy())

            print(classification_report(all_labels, all_predictions))
            report_dict = classification_report(all_labels, all_predictions, output_dict=True)
            save_classification_report("HEAT", report_dict, "./result/HEAT.csv")
            model.eval()


if __name__ == '__main__':
    print(datetime.datetime.now(), 'start train')
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, required=True)
    parser.add_argument('--hidden_channels', type=int, default=64)
    parser.add_argument('--num_layers', type=int, default=4)
    parser.add_argument('--num_heads', type=int, default=1)
    parser.add_argument('--leakage', type=str, default=True)
    parser.add_argument('--mapping', type=str, default=True)
    parser.add_argument('--gpu', type=str, default=True)
    parser.add_argument('--k_folds', type=int, default=5)
    parser.add_argument('--epoch', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--weight_decay', type=float, default=5e-4)
    parser.add_argument('--p_norm', type=int, default=4)
    args = parser.parse_args()

    train(
        data_path=args.data_path,
        model_args=dict(
            hidden_channels=args.hidden_channels,
            num_layers=args.num_layers
        ), **{
            'lr': args.lr,
            'weight_decay': args.weight_decay,
            'epoch': args.epoch,
            'batch_size': args.batch_size,
            'leakage': args.leakage,
            'mapping': args.mapping,
            'k_folds': args.k_folds,
            'p_norm': args.p_norm,
            'gpu': args.gpu,
        }
    )
    print(datetime.datetime.now(), 'end train')
