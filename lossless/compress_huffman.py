import numpy as np
import heapq
from collections import defaultdict

class HuffmanNode:
    def __init__(self, char=None, freq=None, left=None, right=None):
        self.char = char
        self.freq = freq
        self.left = left
        self.right = right

    def __lt__(self, other):
        return self.freq < other.freq

def build_frequency_dict(data):
    freq_dict = defaultdict(int)
    for char in data:
        freq_dict[char] += 1
    return freq_dict

def build_priority_queue(freq_dict):
    heap = []
    for char, freq in freq_dict.items():
        node = HuffmanNode(char, freq)
        heapq.heappush(heap, node)
    return heap

def build_huffman_tree(heap):
    while len(heap) > 1:
        left = heapq.heappop(heap)
        right = heapq.heappop(heap)
        merged = HuffmanNode(freq=left.freq + right.freq, left=left, right=right)
        heapq.heappush(heap, merged)
    return heapq.heappop(heap)

def generate_huffman_codes(root):
    codes = {}
    def generate_codes_helper(node, current_code):
        if node is not None:
            if node.char is not None:
                codes[node.char] = current_code
            generate_codes_helper(node.left, current_code + "0")
            generate_codes_helper(node.right, current_code + "1")
    generate_codes_helper(root, "")
    return codes

def encode_data(data, huffman_codes):
    encoded_data = ''.join(huffman_codes[char] for char in data)
    encode_data = np.array(list(encoded_data), dtype=np.int8)
    return encode_data

def decode_data(encoded_data, root, data_shape):
    if not isinstance(encoded_data, np.ndarray):
        raise ValueError("Encoded data must be a NumPy array")
    
    decoded_output = []
    current_node = root
    for bit in encoded_data:
        if bit == 0:
            current_node = current_node.left
        else:
            current_node = current_node.right
        
        if current_node.char is not None:
            decoded_output.append(current_node.char)
            current_node = root
    decoded_output = np.array(decoded_output)
    decoded_output = decoded_output.reshape(data_shape)
    return decoded_output

def huffman_encoding(data):
    if not isinstance(data, np.ndarray):
        raise ValueError("Input data must be a NumPy array")
    data_shape = data.shape
    data = data.flatten()

    freq_dict = build_frequency_dict(data)
    heap = build_priority_queue(freq_dict)
    root = build_huffman_tree(heap)
    huffman_codes = generate_huffman_codes(root)
    encoded_data = encode_data(data, huffman_codes)
    return encoded_data, root, data_shape
