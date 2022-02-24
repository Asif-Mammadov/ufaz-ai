from entropy import Entropy
from node import Node


string = "peter piper picked a peck of pickled peppers"
e = Entropy(string)

print("Entropy:", e.compute_entropy())
print(e.dict)
print()
nodes = []
for key in e.dict:
  # print(e.dict[key])
  nodes.append(Node(key, freq=e.dict[key]))


nodes.sort(key=lambda x: x.freq)

while len(nodes) > 1:
  nodes.sort(key=lambda x: x.freq)
  least = nodes[:2]
  left, right = least
  newNode = Node(left.data + right.data, left.freq + right.freq, left, right)
  nodes.append(newNode)
  nodes.remove(left)
  nodes.remove(right)

nodes[0].display_propagate(nodes[0])