import random

def generate_text(model, vocab, max_len=20):
    context = ["<s>", "<s>", "<s>"]
    result = []

    for _ in range(max_len):
        probs = [(w, model.prob(w, context)) for w in vocab]
        words, weights = zip(*probs)

        next_word = random.choices(words, weights)[0]

        if next_word == "</s>":
            break

        result.append(next_word)
        context.append(next_word)

    return " ".join(result)
