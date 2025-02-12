from collections import defaultdict
from typing import List, Tuple, TypeVar, Generic, Dict
import heapq
from dataclasses import dataclass

from codecs_funcs.encoding_functions import BitStreamEncoder

Elem = TypeVar('Elem')

# <editor-fold desc=" ------------------------ Nodes for the binary trees ------------------------">
class Node(Generic[Elem]):

    def as_dict(self) -> Tuple[Dict[str, str], str]:
        nodes = dict()

        def dfs(root: Node, index: int) -> Tuple[str, int]:
            if isinstance(root, Leaf):
                return str(root.data), index
            left_node, index = dfs(root.left, index)
            right_node, index = dfs(root.right, index)
            nodes[f'_{index}'] = (left_node, right_node)
            return f'_{index}', index+1
        root, _ = dfs(self, 0)
        return nodes, root

@dataclass(frozen=True)
class Branch(Node[Elem]):
    left: Node[Elem]
    right: Node[Elem]

@dataclass(frozen=True)
class Leaf(Node[Elem]):
    data: Elem

# </editor-fold>

class HuffmanFactory:

    @staticmethod
    def tree_from_list(elements: List[Elem]):
        # compute frequencies
        counters = defaultdict(lambda: 0)
        total = len(elements)
        for elem in elements:
            counters[elem] += 1
        frequency = {symbol: value/total for symbol, value in counters.items()}

        # freq_list = sorted([(freq, symbol) for symbol, freq in frequency.items()])
        # for freq, symbol in freq_list:
        #     print(f'{chr(int(symbol, 2))} : {freq}   -   {counters[symbol]}')
        # print(total)

        # Generate the Huffman tree. The middle index is just so that the heap will not try to
        # sort the node objects (in case two of them have the same weight).
        nodes: List[Tuple[float, int, Node[Elem]]] = [
            (weight, -i, Leaf(data=elem)) for i, (elem, weight) in enumerate(frequency.items())]
        heapq.heapify(nodes)  # min heap

        index = 1
        while len(nodes) > 1:
            # combine smallest two nodes
            weight_left, _, node_left = heapq.heappop(nodes)
            weight_right, _, node_right = heapq.heappop(nodes)
            heapq.heappush(
                nodes, (weight_left + weight_right, index, Branch(left=node_left, right=node_right)))
            index += 1

        return nodes[0][2]

class BinaryTreeRepEncoder(BitStreamEncoder[Node[str]]):

    def __init__(self, token_len: int):
        self.token_length = token_len

    def encode(self, root: Node) -> str:

        def dfs(node: Node[Elem]) -> str:
            if isinstance(node, Leaf):
                assert len(node.data) == self.token_length, f'Length of \'{node.data}\' must be {self.token_length}'
                return '1'+node.data

            # should be a branch node
            if isinstance(node, Branch):
                return f'0{dfs(node.left)}{dfs(node.right)}'

            raise Exception('Node should be either a Lead or a Branch')

        return dfs(root)

    def decode_bits(self, bit_stream: str, index: int) -> Tuple[Node, int]:

        def dfs(bit_stream: str, index: int) -> Tuple[Node[str], int]:
            if bit_stream[index] == '1':
                return Leaf(data=bit_stream[index+1: index+1+self.token_length]), index+1+self.token_length

            # should be a branch node
            left_node, index = dfs(bit_stream, index+1)
            right_node, index = dfs(bit_stream, index)
            return Branch(left=left_node, right=right_node), index

        return dfs(bit_stream, index)

class BinaryTreeEncoder(BitStreamEncoder[Elem]):
    def __init__(self, root: Node[Elem]):

        self.encoding = dict()
        self.root = root

        def dfs(node: Node[Elem], path: str):
            if isinstance(node, Leaf):
                self.encoding[node.data] = path
                return

            if isinstance(node, Branch):
                dfs(node.left, path+'0')
                dfs(node.right, path+'1')
                return

            raise Exception('Node must be a leaf or a branch')

        dfs(self.root, '')


    def encode(self, source: str) -> str:
        return self.encoding[source]

    def decode_bits(self, bit_stream: str, index: int) -> Tuple[str, int]:
        node = self.root
        while isinstance(node, Branch):
            node = node.left if int(bit_stream[index], 2) == 0 else node.right
            index += 1
        return node.data, index


