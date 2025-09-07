import random

with open("/workspace/dataset/corpus_olaf_orig.txt") as f:
    blocks = f.read().split("\n\n")  # paragraph split

random.shuffle(blocks)

with open("/workspace/dataset/corpus_olaf.txt", "w") as f:
    f.write("\n\n".join(blocks))
