class BackoffLM:
    def __init__(self, uni, bi, tri, four):
        self.uni = uni
        self.bi = bi
        self.tri = tri
        self.four = four

    def prob(self, word, context):
        # 4-gram
        if len(context) >= 3:
            key = tuple(context[-3:] + [word])
            if key in self.four:
                total = sum(v for k, v in self.four.items() if k[:-1] == key[:-1])
                return self.four[key] / total

        # 3-gram
        if len(context) >= 2:
            key = tuple(context[-2:] + [word])
            if key in self.tri:
                total = sum(v for k, v in self.tri.items() if k[:-1] == key[:-1])
                return self.tri[key] / total

        # 2-gram
        if len(context) >= 1:
            key = tuple(context[-1:] + [word])
            if key in self.bi:
                total = sum(v for k, v in self.bi.items() if k[:-1] == key[:-1])
                return self.bi[key] / total

        # unigram
        return self.uni.get((word,), 0) / sum(self.uni.values())
