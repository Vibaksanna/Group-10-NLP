class InterpolationLM:
    def __init__(self, uni, bi, tri, four, lambdas, k, vocab_size):
        self.uni = uni
        self.bi = bi
        self.tri = tri
        self.four = four
        self.l = lambdas
        self.k = k
        self.V = vocab_size

    def add_k(self, count, total):
        return (count + self.k) / (total + self.k * self.V)

    def prob(self, word, context):
        # unigram
        p1 = self.add_k(self.uni.get((word,), 0), sum(self.uni.values()))

        # bigram
        p2 = 0
        if len(context) >= 1:
            total = sum(v for k, v in self.bi.items() if k[0] == context[-1])
            p2 = self.add_k(self.bi.get((context[-1], word), 0), total)

        # trigram
        p3 = 0
        if len(context) >= 2:
            total = sum(v for k, v in self.tri.items() if k[:2] == tuple(context[-2:]))
            p3 = self.add_k(self.tri.get((context[-2], context[-1], word), 0), total)

        # 4-gram
        p4 = 0
        if len(context) >= 3:
            total = sum(v for k, v in self.four.items() if k[:3] == tuple(context[-3:]))
            p4 = self.add_k(self.four.get(tuple(context[-3:] + [word]), 0), total)

        return (self.l[0]*p4 + self.l[1]*p3 +
                self.l[2]*p2 + self.l[3]*p1)
