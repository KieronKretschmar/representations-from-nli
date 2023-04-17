class Vocabulary():
    def __init__(self, unk_token = "<UNK>"):
        self.w2i = {}
        self.i2w = []
        self.size = 0
        self.unk_token = unk_token

        self.add_word(self.unk_token)
        self.unk_idx = self.size

    def add_word(self, word):
        if word not in self.w2i:
            self.w2i[word] = self.size
            self.i2w.append(word)
            self.size += 1