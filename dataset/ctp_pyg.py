import torch
import os
import json
import networkx as nx
import csv
import datetime
import hashlib
import time
import multiprocessing
import functools

from collections import defaultdict
from functools import lru_cache
from torch_geometric.data import InMemoryDataset
from torch_geometric.data import HeteroData
from utils.embedding import text_tokenizing
from typing import Tuple, Callable, List

@lru_cache(maxsize=32)
def _load_data(fn):
    return torch.load(fn)


def pad_tensor(tensor, target_size, dim=1):
    padding_size = target_size - tensor.size(dim)
    if padding_size > 0:
        padding = torch.zeros(tensor.size(0), padding_size, device=tensor.device)
        tensor = torch.cat((tensor, padding), dim=dim)
    return tensor


class CrossChainPairDataset(InMemoryDataset):
    def __init__(
            self,
            root: str,
            slice_len: int = 100,
            currency: int = 5,
            transform: Callable = None,
    ):
        self.slice_len = slice_len
        self._currency_lock = multiprocessing.Semaphore(currency)
        super().__init__(root, transform)

    @property
    def raw_file_names(self):
        file_path = os.join(self.root, 'raw')
        file_names = [filename for filename in os.listdir(file_path) if
                      os.path.isfile(os.path.join(file_path, filename))]
        return file_names

    @property
    def processed_file_names(self):
        return ['index.csv', 'metadata.json']

    @property
    def data_index(self) -> List:
        if not getattr(self, '_data_index', None):
            self._data_index = list()
            path = os.path.join(self.processed_dir, 'index.csv')
            with open(path, 'r', encoding='utf-8') as f:
                for row in csv.DictReader(f):
                    self._data_index.append({
                        'filename': row['filename'],
                        'offset': row['offset'],
                    })
        return self._data_index

    @property
    def metadata(self) -> Tuple:
        if not getattr(self, '_metadata', None):
            path = os.path.join(self.processed_dir, 'metadata.json')
            with open(path, 'r') as f:
                data = json.load(f)
            self._metadata = data[0], [tuple(item) for item in data[1]]
        return self._metadata

    def len(self) -> int:
        return len(self.data_index)

    def get(self, idx: int) -> HeteroData:
        info = self.data_index[idx]
        path = os.path.join(self.processed_dir, info['filename'])
        data_list = _load_data(path)
        return data_list[int(info['offset'])]

    def process(self):
        print(datetime.datetime.now(), 'start processing')
        signal = multiprocessing.Semaphore(0)
        for path in os.listdir(self.raw_dir):
            self._currency_lock.acquire()
            path = os.path.join(self.raw_dir, path)
            func = functools.partial(
                CrossChainPairDataset._save_transaction_graph,
                path,
                self._currency_lock,
                signal,
                self.processed_dir,
                self.slice_len,
            )
            p = multiprocessing.Process(target=func)
            p.start()
            print(datetime.datetime.now(), 'processing:', path)

        for _ in range(len(os.listdir(self.raw_dir))):
            signal.acquire()

        print('build index and metadata...')
        index_path = os.path.join(self.processed_dir, 'index.csv')
        index_file = open(index_path, 'w', encoding='utf-8', newline='\n')
        index_writer = csv.writer(index_file)
        index_writer.writerow(['index', 'filename', 'offset'])
        index, node_types, edge_types = 0, set(), set()

        for fn in os.listdir(self.processed_dir):
            if not fn.endswith('.pt'):
                continue
            path = os.path.join(self.processed_dir, fn)
            for offset, data in enumerate(torch.load(path)):
                index_writer.writerow([index, fn, offset])
                index += 1
                node_types.update(data.node_types)
                edge_types.update(data.edge_types)

        index_file.close()
        path = os.path.join(self.processed_dir, 'metadata.json')
        with open(path, 'w') as f:
            json.dump([list(node_types), list(edge_types)], f)

    @staticmethod
    def _save_transaction_graph(
            path: str,
            currency_lock: multiprocessing.Semaphore,
            signal: multiprocessing.Semaphore,
            processed_dir: str,
            slice_len: int
    ):
        data_list = list()

        txpair2label = dict()
        label_path = os.path.join(path, 'label.csv')
        if os.path.exists(label_path):
            with open(label_path, 'r', encoding='utf-8') as f:
                for row in csv.DictReader(f):
                    tx_pair = row['SrcTxHash'] + '@' + row['DstTxHash']
                    txpair2label[tx_pair] = row['TxPairLabel']

        for file_name in os.listdir(path):
            if file_name.endswith('.pt'):
                file_full_path = os.path.join(path, file_name)
                tx_pair = file_name.split('.')[0]
                data = torch.load(file_full_path)
                G = CrossChainPairDataset.process_transaction_graph(data)

                G['transaction_pair'] = tx_pair
                if txpair2label.get(tx_pair):
                    G['label'] = txpair2label[tx_pair]
                else:
                    continue
                data_list.append(G)

                if len(data_list) < slice_len:
                    continue
                fn = hashlib.sha256(str(time.time()).encode()).hexdigest()
                processed_path = os.path.join(processed_dir, '{}.pt'.format(fn))
                torch.save(data_list, processed_path)
                data_list = list()

        if len(data_list) > 0:
            fn = hashlib.sha256(str(time.time()).encode()).hexdigest()
            processed_path = os.path.join(processed_dir, '{}.pt'.format(fn))
            torch.save(data_list, processed_path)

        currency_lock.release()
        signal.release()

    @staticmethod
    def process_transaction_graph(graph: nx.DiGraph) -> HeteroData:
        data = HeteroData()

        node_type2feats = dict()
        node_type2nodes = dict()
        for node, attrs in graph.nodes(data=True):
            if 'type' in attrs.keys():
                t = attrs['type']
            else:
                continue
            if node_type2feats.get(t) is None:
                node_type2feats[t] = list()
                node_type2nodes[t] = list()
            feats = [graph.in_degree(node), graph.out_degree(node)]

            if t == 'Log':
                feats.append(str(attrs['event_name']))
            node_type2feats[t].append(feats)
            node_type2nodes[t].append(node)

        node2idx = dict()
        for nodes in node_type2nodes.values():
            for i, node in enumerate(nodes):
                node2idx[node] = i
        edge_type2edges = dict()

        for u, v, attrs in graph.edges(data=True):
            t = graph.nodes[u].get('type'), attrs['type'], graph.nodes[v].get('type')
            if edge_type2edges.get(t) is None:
                edge_type2edges[t] = list()
            edge_type2edges[t].append([node2idx[u], node2idx[v]])

        edge_type2edge_attr = dict()
        for u, v, attrs in graph.edges(data=True):
            t = graph.nodes[u].get('type'), attrs['type'], graph.nodes[v].get('type')
            if edge_type2edge_attr.get(t) is None:
                edge_type2edge_attr[t] = list()
            if attrs['type'] == 'Transaction':
                edge_type2edge_attr[t].append([
                    *[float(num) for num in ('%e' % attrs.get('value', -1)).split('e')],
                    *[float(num) for num in ('%e' % attrs.get('gas', -1)).split('e')],
                    *[float(num) for num in ('%e' % attrs.get('gas_price', -1)).split('e')],
                    *[float(num) for num in ('%e' % attrs.get('transaction_index', -1)).split('e')],
                    *[float(num) for num in ('%e' % attrs.get('block_number', -1)).split('e')],
                    int(attrs.get('is_create_contract', False)),
                    int(attrs.get('is_error', False)),
                    str(attrs.get('func_name', '')),
                ])

            elif attrs['type'] == 'Emit':
                edge_type2edge_attr[t].append([
                    *[float(num) for num in ('%e' % attrs.get('emit_index', -1)).split('e')],
                    int(attrs.get('removed', False)),
                ])
            elif attrs['type'] == 'Transfer':
                edge_type2edge_attr[t].append([
                    *[float(num) for num in ('%e' % attrs.get('log_index', -1)).split('e')],
                    *[float(num) for num in ('%e' % attrs.get('total_supply', -1)).split('e')],
                    str(attrs.get('name', '')),
                ])
            elif attrs['type'] == 'Token20Transfer':
                edge_type2edge_attr[t].append([
                    *[float(num) for num in ('%e' % attrs.get('log_index', -1)).split('e')],
                    *[float(num) for num in ('%e' % attrs.get('value', -1)).split('e')],
                    *[float(num) for num in ('%e' % attrs.get('total_supply', -1)).split('e')],
                    str(attrs.get('name', '')),
                ])
            elif attrs['type'] == 'Token721Transfer':
                edge_type2edge_attr[t].append([
                    *[float(num) for num in ('%e' % attrs.get('log_index', -1)).split('e')],
                    *[float(num) for num in ('%e' % attrs.get('total_supply', -1)).split('e')],
                    str(attrs.get('name', '')),
                ])
            elif attrs['type'] == 'Token1155Transfer':
                edge_type2edge_attr[t].append([
                    *[float(num) for num in ('%e' % attrs.get('log_index', -1)).split('e')],
                    *[float(num) for num in ('%e' % attrs.get('value', -1)).split('e')],
                    *[float(num) for num in ('%e' % attrs.get('decimals', -1)).split('e')],
                    *[float(num) for num in ('%e' % attrs.get('total_supply', -1)).split('e')],
                    str(attrs.get('name', '')),
                ])
            elif attrs['type'] == 'TokenApproval':
                edge_type2edge_attr[t].append([
                    *[float(num) for num in ('%e' % attrs.get('log_index', -1)).split('e')],
                    *[float(num) for num in ('%e' % attrs.get('value', -1)).split('e')],
                    *[float(num) for num in ('%e' % attrs.get('decimals', -1)).split('e')],
                    *[float(num) for num in ('%e' % attrs.get('total_supply', -1)).split('e')],
                    str(attrs.get('name', '')),
                ])
            elif attrs['type'] == 'TokenApprovalAll':
                edge_type2edge_attr[t].append([
                    *[float(num) for num in ('%e' % attrs.get('log_index', -1)).split('e')],
                    *[float(num) for num in ('%e' % attrs.get('total_supply', -1)).split('e')],
                    int(attrs.get('approved', False)),
                    str(attrs.get('name', '')),
                ])
            else:
                edge_type2edge_attr[t].append([
                    *[float(num) for num in ('%e' % attrs.get('index', -1)).split('e')],
                    *[float(num) for num in ('%e' % attrs.get('value', -1)).split('e')],
                    *[float(num) for num in ('%e' % attrs.get('gas', -1)).split('e')],
                ])

        for t, feats in node_type2feats.items():
            if isinstance(feats[0][-1], str):
                texts = [feat[-1] for feat in feats]
                text_feats = text_tokenizing(texts)
                for i, feat in enumerate(feats):
                    feat.pop(-1)
                    feat.extend(text_feats[i])
            data[t].x = torch.tensor(feats)
        for t, edge_attrs in edge_type2edge_attr.items():
            if isinstance(edge_attrs[0][-1], str):
                texts = [edge_attr[-1] for edge_attr in edge_attrs]
                text_feats = text_tokenizing(texts)
                for i, edge_attr in enumerate(edge_attrs):
                    edge_attr.pop(-1)
                    edge_attr.extend(text_feats[i])
            data[t].edge_attr = torch.tensor(edge_attrs)
        for t, edges in edge_type2edges.items():
            data[t].edge_index = torch.tensor(edges).t().contiguous()

        return data


if __name__ == '__main__':
    dataset = CrossChainPairDataset(
        root='BridgeShield/pair_data'
    )
