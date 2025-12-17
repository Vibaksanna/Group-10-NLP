import random
from collections import Counter

def load_corpus(path):
    with open(path, "r", encoding="utf-8") as f:
        text = f.read().lower()
    return text.split()

def split_data(tokens):
    random.shuffle(tokens)
    n = len(tokens)

    train = tokens[:int(0.7 * n)]
    valid = tokens[int(0.7 * n):int(0.8 * n)]
    test  = tokens[int(0.8 * n):]

    return train, valid, test

def build_vocab(tokens, vocab_size=5000):
    freq = Counter(tokens)
    vocab = set([w for w, _ in freq.most_common(vocab_size)])
    vocab.add("<UNK>")
    return vocab

def replace_unk(tokens, vocab):
    return [w if w in vocab else "<UNK>" for w in tokens]
