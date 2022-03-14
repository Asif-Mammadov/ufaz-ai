from entropy import Entropy
from node import Node
from huffman import Huffman

string = "peter piper picked a peck of pickled peppers"
e = Entropy(string)

print("Entropy:", e.compute_entropy())
print("Count: ", e.get_count())
print()

huffman = Huffman(e)

huffman.bfs_print()
dict = huffman.get_encoded()
print(dict)

print(huffman.encoding_of('d'))