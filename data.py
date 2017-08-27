import os
import torch
import torch.autograd as autograd



class Dictionary(object):
    def __init__(self):
        # working with word
        self.word2idx = {}
        self.idx2word = []

        # working with tag
        self.tag2idx = {}
        self.idx2tag = []

    def add_word(self, word):
        if word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1
        return self.word2idx[word]

    def add_tag(self, tag):
        if tag not in self.tag2idx:
            self.idx2tag.append(tag)
            self.tag2idx[tag] = len(self.idx2tag) - 1
        return self.tag2idx[tag]

    def __len__(self):
        return len(self.idx2word)


class Corpus(object):
    def __init__(self, path):
        self.dictionary = Dictionary()
        self.train = self.tuple_sentence(os.path.join(path, 'train.txt'))
        self.valid = self.tuple_sentence(os.path.join(path, 'valid.txt'))
        self.test = self.tuple_sentence(os.path.join(path, 'test.txt'))

    def tuple_sentence(self, path):
        """Tokenizes a text file."""
        print('path =' + path)
        assert os.path.exists(path)
        # Add words to the dictionary
        data = []
        with open(path, 'r') as f:
            for line in f:
                pair_word_tags = line.split()
                words = []
                tags = []
                for pair in pair_word_tags:
                    if '\\' in pair:
                        word, tag = pair.split('\\')
                        words.append(word)
                        tags.append(tag)
                        self.dictionary.add_word(word)
                        self.dictionary.add_tag(tag)
                    else:
                        print(pair)
                data.append((words, tags))

        return data
        # # Tokenize file content
        # with open(path, 'r') as f:
        #     ids = torch.LongTensor(tokens)
        #     token = 0
        #     for line in f:
        #         words = line.split() + ['<eos>']
        #         for word in words:
        #             ids[token] = self.dictionary.word2idx[word]
        #             token += 1
        #
        # return ids

    def prepare_sequence(self, seq, to_ix, sequence_has_tag = False):
        if sequence_has_tag:
            words = []
            for pair in seq:
                if '\\' in pair:
                    word, tag = pair.split('\\')
                    words.append(word)
            seq = words

        idxs = [to_ix[w] for w in seq]
        tensor = torch.LongTensor(idxs)
        return autograd.Variable(tensor)