import os
import pickle
import random
import logging

import numpy as np

# 2600+
from gensim.models.keyedvectors import KeyedVectors
unk = [0.0123974,-0.0161108,0.113953,-0.0206721,-0.296849,0.0333717,-0.116066,0.0284888,0.0632764,-0.186654,0.198107,0.0373734,0.094592,0.0897812,-0.209417,-0.107555,0.228596,-0.112332,0.00629923,-0.0917881,0.0772907,0.0333546,-0.108149,0.0395139,-0.0495847,0.115166,0.00213929,0.157468,-0.0235416,-0.0440023,-0.0519611,0.0741484,0.118609,-0.197312,0.0977103,0.0662098,0.291297,0.115577,-0.140065,0.0835127,-0.537392,0.119384,-0.076522,0.0954565,-0.0660837,0.0571342,-0.00752284,-0.0320966,0.112971,0.0083828,-0.199773,-0.0473039,0.0103183,0.150188,-0.407051,-0.0313375,-0.205304,0.0701054,-0.0809467,-0.0403719,-0.38838,0.168263,-0.0719462,-0.265293,0.0604563,0.0329507,0.0593362,0.00510517,-0.0301253,-0.0257452,0.154013,-0.392062,0.147688,0.141134,-0.233924,-0.0516733,0.0430464,0.0411127,-0.00644766,-0.016719,0.182182,0.0292113,0.469613,0.0172282,0.141538,-0.0989069,-0.163462,-0.0959468,-0.0207545,0.327248,0.165677,-0.0178294,0.330997,0.0421784,0.0865335,0.0339321,-0.129773,-0.0282569,-0.00990369,-0.256167]
def load_word2vec_bin(path, word_to_id):
    word_invec = 0
    word_vec = KeyedVectors.load(path)
    word2vec = []
    for word, i in word_to_id.items():
        if word in word_vec:
            word2vec.append(word_vec[word])
            word_invec += 1
        else:
            word2vec.append(unk)
    print ('find ' + str(word_invec) + ' in word2vec.')
    print ('total words ' + str(len(list(word_to_id.keys()))))
    print ('totoal w2v ' + str(len(word_vec.vocab)))
    return word2vec

#  4300+ / 45oo
# word2vec [[1,2], [], []] vocab
def load_word2vec(path, word_to_id):
    word_invec = 0
    with open(path, "rb") as f:
        word_vec = pickle.load(f)
        word2vec = []
        for word, i in word_to_id.items():
            if word in word_vec:
                word2vec.append(word_vec[word])
                word_invec += 1
            else:
                word2vec.append(word_vec["<UNK>"])
    print ('find ' + str(word_invec) + ' in word2vec.')
    print ('total words ' + str(len(list(word_to_id.keys()))))
    print ('totoal w2v ' + str(len(list(word_vec))))
    return word2vec


def get_logger(name):
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    fh = logging.FileHandler(os.path.join("./log", name + ".log"))
    fh.setLevel(logging.DEBUG)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    ch.setFormatter(formatter)
    fh.setFormatter(formatter)
    logger.addHandler(ch)
    logger.addHandler(fh)
    return logger

def write_test(results, path):
    output_file = os.path.join(path, "ner_predict.utf8")
    with open(output_file, "w") as f:
        to_write = []
        for block in results:
            for line in block[0]:
                to_write.append(line + "\n")
            if block[1]:
                to_write.append("\n")

        f.writelines(to_write)

def test_ner(results, path):
    script_file = "./conlleval"
    output_file = os.path.join(path, "ner_predict.utf8")
    result_file = os.path.join(path, "ner_result.utf8")
    with open(output_file, "w") as f:
        to_write = []
        for block in results:
            for line in block[0]:
                to_write.append(line + "\n")
            if block[1]:
                to_write.append("\n")

        f.writelines(to_write)
    os.system("perl {} < {} > {}".format(script_file, output_file, result_file))
    eval_lines = []
    with open(result_file) as f:
        for line in f:
            eval_lines.append(line.strip())
            print (line.strip())
    return eval_lines


def calculate_accuracy(labels, paths, lengths):
    # calculate token level accuracy, return correct tag numbers and total tag numbers
    total = 0
    correct = 0
    for label, path, length in zip(labels, paths, lengths):
        gold = label[length]
        correct += np.sum(np.equal(gold, path))
        total += length
    return correct, total


class BatchManager(object):

    def __init__(self, data, num_tag, word_max_len, batch_size):
        self.data = data
        self.numbatch = len(self.data) // batch_size
        self.batch_size = batch_size
        self.batch_index = 0
        self.len_data = len(data)
        self.num_tag = num_tag

    @staticmethod
    def unpack(data):
        words = []
        tags = []
        lengths = []
        features = []
        str_lines = []
        end_of_doc = []
        for item in data:
            if item["len"] < 0:
                continue
            words.append(item["words"])
            tags.append(item["tags"])
            lengths.append(item["len"])
            features.append(item["features"])
            str_lines.append(item["str_line"])
            end_of_doc.append(item["end_of_doc"])
        return {"words": words,
                "tags": tags,
                "len": lengths,
                "features": features,
                "str_lines": str_lines,
                "end_of_doc": end_of_doc}

    def shuffle(self):
        random.shuffle(self.data)

    def iter_batch(self):
        for i in range(self.numbatch+1):
            if i == self.numbatch:
                data = self.data[i*self.batch_size:]
            else:
                data = self.data[i*self.batch_size:(i+1)*self.batch_size]
            yield self.unpack(data)

