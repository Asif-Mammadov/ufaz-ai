from entropy import Entropy
from node import Node
from huffman import Huffman

string = "peter piper picked a peck of pickled peppers"
e = Entropy(string)

print("Entropy:", e.compute_entropy())
print("Count: ", e.get_count())
print()

huffman = Huffman(e)

print("Huffman Tree")
huffman.bfs_print()
dict = huffman.get_encoded()
print(dict)

encoded_str = huffman.to_encoding(string)
print("Encoded string:\n{}".format(encoded_str))

print("Decoded string:\n{}".format(huffman.to_string(encoded_str)))
