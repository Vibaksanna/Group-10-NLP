from src.preprocess import *
from src.ngram import *
from src.lm_backoff import BackoffLM
from src.lm_interpolation import InterpolationLM
from src.perplexity import perplexity
from src.generate import generate_text

# Load & preprocess
tokens = load_corpus("data/corpus.txt")
train, valid, test = split_data(tokens)

vocab = build_vocab(train, 5000)
train = replace_unk(train, vocab)
test = replace_unk(test, vocab)

# Build n-grams
uni = build_ngrams(train, 1)
bi  = build_ngrams(train, 2)
tri = build_ngrams(train, 3)
four= build_ngrams(train, 4)

# Models
lm1 = BackoffLM(uni, bi, tri, four)
lm2 = InterpolationLM(
    uni, bi, tri, four,
    lambdas=[0.4, 0.3, 0.2, 0.1],
    k=1,
    vocab_size=len(vocab)
)

# Evaluation
print("Perplexity LM1:", perplexity(lm1, test))
print("Perplexity LM2:", perplexity(lm2, test))

# Generation
print("\nLM1 Generated Text:")
print(generate_text(lm1, vocab))

print("\nLM2 Generated Text:")
print(generate_text(lm2, vocab))
