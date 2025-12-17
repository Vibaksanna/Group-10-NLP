from collections import defaultdict

def build_ngrams(tokens, n):
    ngrams = defaultdict(int)
    padded = ["<s>"] * (n - 1) + tokens + ["</s>"]

    for i in range(len(padded) - n + 1):
        ngram = tuple(padded[i:i + n])
        ngrams[ngram] += 1

    return ngrams
