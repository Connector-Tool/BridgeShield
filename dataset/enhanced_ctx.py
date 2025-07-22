from typing import Iterator, Tuple
from daos import EventLogReader, TransactionReader, TraceReader, \
    Token20TransferReader, Token721TransferReader, Token1155TransferReader, \
    TokenApprovalReader, TokenApprovalAllReader, JointReader
from utils.signature import load_signatures
from utils.router import load_router_contract
from utils.token import load_token_contract
import networkx as nx
from collections import Counter

class EnCrossChainTx:
    def __init__(self, root: str, signature_path: str, router_path: str, token_path: str, chain: str, label:str):
        # label是表明是源链还是目标链交易
        self.root = root
        self.signature2keyword = load_signatures(signature_path)
        self.router_contract_list = load_router_contract(router_path)
        self.token_contract_list = load_token_contract(token_path)
        self.chain = chain # 表明这笔交易是属于什么链的
        self.label = label

    def iter_read(self) -> Iterator[Tuple[str, nx.MultiDiGraph]]:
        yield from self._load_transaction_graphs(path=self.root)

    def get_node_type(self, address) -> str:
        address = address.lower()
        if address in self.token_contract_list:
            return 'Token Contract'
        else:
            return 'Other'

    def _load_transaction_graphs(self, path: str) -> Iterator[nx.MultiDiGraph]:
        tx2graph = dict()
        tx2top_func_name = dict()

        reader = TransactionReader(path, signature2keyword=self.signature2keyword)
        reader = JointReader(
            path, 'TransactionReceiptItem.csv',
            joint_reader=reader,
            joint_key='transaction_hash'
        )
        for item in reader.iter_read():
            g = nx.MultiDiGraph()
            address_from = item['address_from'].lower()
            if isinstance(item['address_to'],float):
                address_to = 'self_destruct'
            else:
                address_to = item['address_to'].lower()

            address_from_key = address_from + '_' + self.chain
            address_to_key = address_to + '_' + self.chain
            if self.label == 'src':
                g.add_node(address_from_key,
                           type='User',
                           total_sent_value=0,
                           total_tx_count=0,
                           failed_tx_count=0,
                           first_seen_timestamp=int(item['timestamp']),
                           last_seen_timestamp=int(item['timestamp'])
                           )

                g.add_node(address_to_key,
                           type="Router Contract",
                           total_received_value=0,
                           total_tx_count=0,
                           failed_tx_count=0,
                           first_seen_timestamp=int(item['timestamp']),
                           last_seen_timestamp=int(item['timestamp'])
                           )

            else:
                g.add_node(address_from_key,
                           type="Router Contract",
                           total_sent_value=0,
                           total_tx_count=0,
                           failed_tx_count=0,
                           first_seen_timestamp=int(item['timestamp']),
                           last_seen_timestamp=int(item['timestamp'])
                           )

                g.add_node(address_to_key,
                           type=self.get_node_type(address_to),
                           total_received_value=0,
                           total_tx_count=0,
                           failed_tx_count=0,
                           first_seen_timestamp=int(item['timestamp']),
                           last_seen_timestamp=int(item['timestamp'])
                           )

            g.nodes[address_from_key]['total_sent_value'] += int(item['value'])
            g.nodes[address_to_key]['total_received_value'] += int(item['value'])
            g.nodes[address_from_key]['total_tx_count'] += 1
            g.nodes[address_to_key]['total_tx_count'] += 1

            if item.get('is_error') == 'True':
                g.nodes[address_from_key]['failed_tx_count'] += 1
                g.nodes[address_to_key]['failed_tx_count'] += 1

            g.nodes[address_from_key]['last_seen_timestamp'] = int(item['timestamp'])
            g.nodes[address_to_key]['last_seen_timestamp'] = int(item['timestamp'])

            g.add_edge(
                address_from_key, address_to_key,
                value=int(item['value']),
                gas=int(item['gas']),
                gas_price=int(item['gas_price']),
                timestamp=int(item['timestamp']),
                block_number=int(item['block_number']),
                is_create_contract=item.get('created_contract') != '',
                is_error=item.get('is_error') == 'True',
                type='Transaction',
                transaction_index=int(item['transaction_index']),
                func_name=item['func_name'],
            )

            tx2graph[item['transaction_hash']] = g
            tx2top_func_name[item['transaction_hash']] = item['func_name']

        reader = TraceReader(path, signature2keyword=self.signature2keyword)
        
        for item in reader.iter_read():
            if g is None:
                continue
            g = tx2graph.get(item['transaction_hash'])
            address_from = item["address_from"].lower()
            address_to = item["address_to"].lower()

            for address in [address_from, address_to]:
                node_key = address + '_' + self.chain
                if not g.has_node(node_key):
                    g.add_node(
                        node_key,
                        type=self.get_node_type(address)
                    )

                g.nodes[node_key].setdefault("total_value", 0)
                g.nodes[node_key].setdefault("total_gas_used", 0)
                g.nodes[node_key].setdefault("tx_count", 0)
                g.nodes[node_key].setdefault("first_seen_timestamp", int(item['timestamp']))
                g.nodes[node_key].setdefault("last_seen_timestamp", int(item['timestamp']))
                g.nodes[node_key].setdefault("func_set", set())

            g.nodes[node_key]["total_value"] += int(item['value'])
            g.nodes[node_key]["total_gas_used"] += int(item['gas_used'])
            g.nodes[node_key]["tx_count"] += 1
            g.nodes[node_key]["last_seen_timestamp"] = int(item['timestamp'])
            g.nodes[node_key]["func_set"].add(",".join(item['func_name']))

            g.add_edge(
                address_from + '_' + self.chain, address_to + '_' + self.chain,
                value=int(item['value']),
                gas=int(item['gas']),
                gas_used=int(item['gas_used']),
                timestamp=int(item['timestamp']),
                block_number=int(item['block_number']),
                trace_id=item["trace_id"],
                type="CALL",
                func_name=item['func_name'],
            )

        for item in EventLogReader(path, signature2keyword=self.signature2keyword).iter_read():
            txhash = item['transaction_hash']
            g = tx2graph.get(txhash)
            if g is None:
                continue
            address = item['address'].lower()
            if not g.has_node(address+ '_' + self.chain):
                g.add_node(
                    address + '_' + self.chain,
                    type = self.get_node_type(address)
                )
            
            log = item['topics']
            topic0 = log[0] if len(log) > 0 else ''
            log_id = '{}@{}'.format(txhash, topic0)
            g.add_node(
                log_id,
                event_name=item['event_name'],
                type='Log',
            )

            g.add_edge(
                address + '_' + self.chain, log_id,
                timestamp=int(item['timestamp']),
                removed=item['removed'] == 'True',
                type='Emit',
            )

        reader = JointReader(
            path, 'TokenPropertyItem.csv',
            joint_reader=Token20TransferReader(path),
            joint_key='contract_address',
        )
        for item in reader.iter_read():
            g = tx2graph.get(item['transaction_hash'])
            if g is None:
                continue

            address_from = item["address_from"].lower()
            address_to = item["address_to"].lower()

            node_key_from = address_from + '_' + self.chain
            if not g.has_node(node_key_from):
                g.add_node(
                    node_key_from,
                    type=self.get_node_type(address_from)
                )

            node_key_to = address_to + '_' + self.chain
            if not g.has_node(node_key_to):
                if self.label == 'dst':
                    g.add_node(node_key_to, type="User")
                else:
                    g.add_node(
                        node_key_to,
                        type=self.get_node_type(address_to)
                    )

            if item['contract_address']:
                g.nodes[node_key_to]["contract_address"] = item['contract_address']
                g.nodes[node_key_to]["name"] = item.get('name', '')
                g.nodes[node_key_to]["token_symbol"] = item.get('token_symbol', '')
                g.nodes[node_key_to]["decimals"] = int(item.get('decimals', -1))
                g.nodes[node_key_to]["total_supply"] = int(float(item.get('total_supply', -1)))

            g.add_edge(
                node_key_from, node_key_to,
                value=int(float(item['value'])),
                type='Transfer',
                log_index=int(item['log_index']),
            )

        reader = JointReader(
            path, 'TokenPropertyItem.csv',
            joint_reader=Token721TransferReader(path),
            joint_key='contract_address',
        )
        for item in reader.iter_read():
            g = tx2graph.get(item['transaction_hash'])
            if g is None:
                continue

            address_from = item["address_from"].lower()
            address_to = item["address_to"].lower()

            node_key_from = address_from + '_' + self.chain
            node_key_to = address_to + '_' + self.chain

            if not g.has_node(node_key_from):
                g.add_node(node_key_from, type=self.get_node_type(address_from))

            if not g.has_node(node_key_to):
                g.add_node(node_key_to, type=self.get_node_type(address_to))

            contract_key = item['contract_address'] + '_' + self.chain
            if not g.has_node(contract_key):
                g.add_node(contract_key, type="Token Contract")

            g.nodes[contract_key]["contract_address"] = item['contract_address']
            g.nodes[contract_key]["name"] = item.get('name', '')
            g.nodes[contract_key]["token_symbol"] = item.get('token_symbol', '')
            g.nodes[contract_key]["total_supply"] = int(float(item.get('total_supply', -1)))

            g.add_edge(
                node_key_from, node_key_to,
                token_id=int(item['token_id']),
                type='Transfer',
                log_index=int(item['log_index']),
            )

        reader = JointReader(
            path, 'TokenPropertyItem.csv',
            joint_reader=Token1155TransferReader(path),
            joint_key='contract_address',
        )
        for item in reader.iter_read():
            g = tx2graph.get(item['transaction_hash'])
            if g is None:
                continue

            address_from = item["address_from"].lower()
            address_to = item["address_to"].lower()

            node_key_from = address_from + '_' + self.chain
            node_key_to = address_to + '_' + self.chain

            if not g.has_node(node_key_from):
                g.add_node(node_key_from, type=self.get_node_type(address_from))

            if not g.has_node(node_key_to):
                g.add_node(node_key_to, type=self.get_node_type(address_to))

            contract_key = item['contract_address'] + '_' + self.chain
            if not g.has_node(contract_key):
                g.add_node(contract_key, type="Token Contract")

            g.nodes[contract_key]["contract_address"] = item['contract_address']
            g.nodes[contract_key]["name"] = item.get('name', '')
            g.nodes[contract_key]["token_symbol"] = item.get('token_symbol', '')
            g.nodes[contract_key]["decimals"] = int(item.get('decimals', -1))
            g.nodes[contract_key]["total_supply"] = int(float(item.get('total_supply', -1)))

            g.add_edge(
                node_key_from, node_key_to,
                token_id=int(item['token_id']),
                value=int(float(item['value'])),
                type='Transfer',
                log_index=int(item['log_index']),
            )

        reader = JointReader(
            path, 'TokenPropertyItem.csv',
            joint_reader=TokenApprovalReader(path),
            joint_key='contract_address',
        )
        for item in reader.iter_read():
            g = tx2graph.get(item['transaction_hash'])
            if g is None:
                continue

            address_from = item["address_from"].lower()
            address_to = item["address_to"].lower()

            node_key_from = address_from + '_' + self.chain
            node_key_to = address_to + '_' + self.chain

            if not g.has_node(node_key_from):
                g.add_node(node_key_from, type=self.get_node_type(address_from))

            if not g.has_node(node_key_to):
                g.add_node(node_key_to, type=self.get_node_type(address_to))

            contract_key = item['contract_address'] + '_' + self.chain
            if not g.has_node(contract_key):
                g.add_node(contract_key, type="Token Contract")

            g.nodes[contract_key]["contract_address"] = item['contract_address']
            g.nodes[contract_key]["name"] = item.get('name', '')
            g.nodes[contract_key]["token_symbol"] = item.get('token_symbol', '')
            g.nodes[contract_key]["decimals"] = int(item.get('decimals', -1))
            g.nodes[contract_key]["total_supply"] = int(float(item.get('total_supply', -1)))

            g.add_edge(
                node_key_from, node_key_to,
                value=int(float(item['value'])),
                type='TokenApproval',
                log_index=int(item['log_index']),
            )

        reader = JointReader(
            path, 'TokenPropertyItem.csv',
            joint_reader=TokenApprovalAllReader(path),
            joint_key='contract_address',
        )
        for item in reader.iter_read():
            g = tx2graph.get(item['transaction_hash'])
            if g is None:
                continue

            address_from = item["address_from"].lower()
            address_to = item["address_to"].lower()

            node_key_from = address_from + '_' + self.chain
            node_key_to = address_to + '_' + self.chain

            if not g.has_node(node_key_from):
                g.add_node(node_key_from, type=self.get_node_type(address_from))

            if not g.has_node(node_key_to):
                g.add_node(node_key_to, type=self.get_node_type(address_to))

            contract_key = item['contract_address'] + '_' + self.chain
            if not g.has_node(contract_key):
                g.add_node(contract_key, type="Token Contract")

            g.nodes[contract_key]["contract_address"] = item['contract_address']
            g.nodes[contract_key]["name"] = item.get('name', '')
            g.nodes[contract_key]["token_symbol"] = item.get('token_symbol', '')
            g.nodes[contract_key]["total_supply"] = int(float(item.get('total_supply', -1)))

            g.add_edge(
                node_key_from, node_key_to,
                approved=bool(item['approved'] == 'True'),
                type='TokenApprovalAll',
                log_index=int(item['log_index']),
            )

        for txhash, g in tx2graph.items():
            yield txhash, g

