from typing import Iterator, Tuple
from dataset.enhanced_ctx import EnCrossChainTx
from pathlib import Path

import shutil
import networkx as nx
import pandas as pd
import torch
import os
import numpy as np

class CrossChainPair:
    def __init__(self, src_path:str, dst_path:str, signature_path, router_path, token_path, label_path):
        self.src_path = src_path
        self.dst_path = dst_path
        self.signature_path = signature_path
        self.router_path = router_path
        self.token_path = token_path
        self.label_path = label_path

    def iter_read(self) -> Iterator[Tuple[str, nx.MultiDiGraph]]:
        yield from self._load_graph()

    def _load_graph(self) -> Iterator[nx.MultiDiGraph]:
        dataset_src = EnCrossChainTx(
            self.src_path,
            signature_path=self.signature_path,
            router_path=self.router_path,
            token_path=self.token_path,
            chain = self.src_path.split('/')[-1],
            label = 'src'
        )
        dataset_dst = EnCrossChainTx(
            self.dst_path,
            signature_path=self.signature_path,
            router_path=self.router_path,
            token_path=self.token_path,
            chain=self.dst_path.split('/')[-1],
            label = 'dst'
        )

        tx_hash_to_graph = {}

        labels_df = pd.read_csv(self.label_path)

        for transaction_hash, transaction_graph in dataset_src.iter_read():
            tx_hash_to_graph[transaction_hash] = transaction_graph

        for transaction_hash, transaction_graph in dataset_dst.iter_read():
            tx_hash_to_graph[transaction_hash] = transaction_graph

        print(len(tx_hash_to_graph))

        merged_graphs = dict()

        for index, row in labels_df.iterrows():
            src_tx_hash = row['SrcTxHash']
            dst_tx_hash = row['DstTxHash']

            if src_tx_hash in tx_hash_to_graph:
                src_graph = tx_hash_to_graph[src_tx_hash]
            else:
                src_graph = None

            if dst_tx_hash in tx_hash_to_graph:
                dst_graph = tx_hash_to_graph[dst_tx_hash]
            else:
                dst_graph = None

            if src_graph is not None and dst_graph is not None:
                G = nx.union(src_graph, dst_graph)
                for node in src_graph.nodes(data='type'):
                    if node[1] == "Router Contract":
                        src_router_contract = node[0]
                        break

                for node in dst_graph.nodes(data='type'):
                    if node[1] == "Router Contract":
                        dst_router_contract = node[0]
                        break

                if src_router_contract and dst_router_contract:
                    G.add_node("Relayer",type="Relayer")
                    G.add_edge(src_router_contract,
                               "Relayer",
                               type='Dummy'
                    )
                    G.add_edge("Relayer",
                               dst_router_contract,
                               type='Dummy'
                    )

                merged_graphs[src_tx_hash + '@' + dst_tx_hash] = G

            elif src_graph is not None or dst_graph is not None:
                if src_graph is not None:
                    G = src_graph
                    for node in src_graph.nodes(data='type'):
                        if node[1] == "Router Contract":
                            src_router_contract = node[0]
                            break
                    G.add_node("Relayer", type="Relayer")
                    G.add_edge(src_router_contract,
                               "Relayer",
                               type='Dummy'
                               )
                    merged_graphs[src_tx_hash + '@'] = G

                if dst_graph is not None:
                    G = dst_graph
                    for node in G.nodes(data='type'):
                        if node[1] == "Router Contract":
                            dst_router_contract = node[0]
                    G.add_node("Relayer", type="Relayer")
                    G.add_edge("Relayer",
                               dst_router_contract,
                               type='Dummy'
                               )
                    merged_graphs['@' + dst_tx_hash] = G
            else:
                print(f"Warning: Both graphs are empty for transaction {src_tx_hash} and {dst_tx_hash}. Creating a default empty graph.")

        for txhash, g in merged_graphs.items():
            yield txhash, g

if __name__ == '__main__':
    PROJECT = 'BSCETH'
    SRC_CHAIN = 'BSC'
    DST_CHAIN = 'ETH'

    # PROJECT = 'CelerBridge'
    # PROJECT = 'ETHBSC'
    # PROJECT = 'Multichain'
    # PROJECT = 'PolyNetwork'
    # SRC_CHAIN = 'ETH'
    # DST_CHAIN = 'BSC'

    label_path = './labels/tx_pair_label/' + PROJECT + '/' + SRC_CHAIN + '_' + DST_CHAIN + '_label.csv'
    src_path = './data/' + PROJECT + '/' + SRC_CHAIN
    dst_path = './data/' + PROJECT +  '/' + DST_CHAIN
    signature_path = './SignItem.csv'
    router_path = './labels/contract_labels/BSC_ETH_bridge_contract.csv'
    token_path = './labels/contract_labels/BSC_ETH_token_contract.csv'


    dataset = CrossChainPair(
        src_path = src_path,
        dst_path = dst_path,
        signature_path = signature_path,
        router_path = router_path,
        token_path = token_path,
        label_path = label_path
    )

    file_path = './pair_data/raw/' + PROJECT + '_' + SRC_CHAIN + '_' + DST_CHAIN + '/'
    if not os.path.exists(file_path):
        os.makedirs(file_path)

    for transaction_pair, transaction_graph in dataset.iter_read():
        file_name = file_path + transaction_pair + '.pt'
        torch.save(transaction_graph, file_name)

    destination_folder = file_path
    destination_path = Path(destination_folder)
    destination_path.mkdir(parents=True, exist_ok=True)
    destination_file_path = destination_path / 'label.csv'

    try:
        shutil.copy2(label_path, destination_file_path)
        print(f"文件已成功复制到 {destination_file_path}")
    except IOError as e:
        print(f"无法复制文件: {e}")
    except Exception as e:
        print(f"发生错误: {e}")
