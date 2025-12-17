import math

def perplexity(model, tokens):
    log_prob = 0
    N = 0

    for i in range(3, len(tokens)):
        context = tokens[i-3:i]
        p = model.prob(tokens[i], context)
        if p > 0:
            log_prob += math.log2(p)
            N += 1

    return 2 ** (-log_prob / N)
