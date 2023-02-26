import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split

torch.manual_seed(1)
random.seed(1)

'''
    Load data Part


'''


def load_data(train_path):
    # data/TREC_10.label.txt
    # data/train_5500.label.txt

    sentences = []
    labels_fine = []
    labels_coarse = []
    with open(train_path, 'r') as f:
        for line in f:
            line_clean = line.strip('\n').split(' ')
            sentence = line_clean[1:-1]
            temp_list = []
            for word in sentence:
                word = word.lower()
                temp_list.append(word)
            fine = line_clean[0]
            coarse = fine.split(':')[0]
            # fine = label.split(':')[1]
            sentences.append(temp_list)
            labels_coarse.append(coarse)
            labels_fine.append(fine)
    print(sentences[0:10])
    print(labels_fine[0:10])
    print(labels_coarse[0:10])
    return sentences, labels_fine, labels_coarse


class QuestionDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        x = self.data[index]
        y = self.labels[index]
        return x, y


def construct_dataset(sentences, labels_fine, labels_coarse, mode):
    if mode == 'coarse':
        dataset = QuestionDataset(sentences, labels_coarse)
        print(dataset)
    else:
        dataset = QuestionDataset(sentences, labels_fine)
        print(dataset)
    #for sentence,label in dataset:

    return dataset


def split_dataset(question_dataset, split_coef):
    train_size = int(split_coef * len(question_dataset))
    dev_size = len(question_dataset) - train_size
    train_dataset, dev_dataset = random_split(question_dataset, [train_size, dev_size])
    print(len(train_dataset))
    print(len(dev_dataset))
    return train_dataset, dev_dataset


'''
    Word Embedding

'''


def construct_dict(vocabs_k):
    dict_vocabs_k = {'#unk#': 0, '#pad#': 1}
    for i, word in enumerate(vocabs_k):
        dict_vocabs_k[word] = i + 2
        # dict_vocabs_k = {word: i for i, word in enumerate(vocabs_k)}
    return dict_vocabs_k


def create_from_random(vocabs, embedding_size):
    dict_random = construct_dict(vocabs)
    print(dict_random)
    embeddings = nn.Embedding(len(dict_random), embedding_size, padding_idx=1)
    return dict_random,embeddings


def create_from_pretrained(glove_path, vocabs , embedding_size):
    labels, embedding = load_glove(glove_path)
    len_labels = len(labels) #9549
    dict_glove,embeddings_glove =prune_glove(labels, embedding,vocabs,embedding_size)
    return dict_glove,embeddings_glove

def prune_glove(labels, embedding,vocabs,embedding_size):
    vocabs.append('#unk#')
    vocabs.append('#pad#')
    new_labels = []
    new_embedding = []
    new_dict = {}
    t = 0
    for i in range(len(labels)):
        if labels[i] in vocabs:
            if labels[i] not in new_dict:
                if t == 1:
                    new_dict['#pad#'] = t

                    new_labels.append('#pad#')
                    temp_list = []
                    for j in range(embedding_size):
                        temp_list.append(0)
                    new_embedding.append(temp_list)
                else:
                    new_dict[labels[i]] = t
                    new_labels.append(labels[i])
                    new_embedding.append(embedding[i][:embedding_size])
                t += 1

    return new_dict,new_embedding



def load_glove(glove_path):
    result = []
    labels = []
    with open(glove_path, 'r') as f:
        for line in f:
            label = line.split('\t')[0].lower()
            back = line.split('\t')[1]
            line_clean = back.strip('\n').split(' ')
            #result.append(line_clean)
            temp_line = []
            for word in line_clean:
                word_float = float(word)
                temp_line.append(word_float)
            result.append(temp_line)
            labels.append(label)
    return labels,result


def create_word_embedding(k, all_sentences, mode, stop_words,model):
    embedding_size = 50
    num_iterations = 100
    all_words = [word for sentence in all_sentences for word in sentence]
    vocabs_k = count_k(all_words, k)
    vocabs_k_noStopWords = eliminate_stop_words(vocabs_k, stop_words=stop_words)
    # sentences = align_sentence(sentences, max_len=5)
    if mode == 'random':
        dict_embedding,embeddings = create_from_random(vocabs_k_noStopWords, embedding_size)
        embeddings_weight = embeddings.weight
    else:
        dict_embedding,embeddings = create_from_pretrained(glove_path, vocabs_k_noStopWords,embedding_size=50)
        embeddings_weight = torch.FloatTensor(embeddings)
    word2vec = produce_word2vec(dict_embedding, embeddings_weight, sentences,max_len = 40,model = model)
    return word2vec

def produce_word2vec(dict_vocab,embeddings_weight,sentences,max_len,model):
    sentences_replaced = replace_with_dict(sentences,dict_vocab)
    sentences_aligned = align_sentence(sentences_replaced, max_len)
    #weight = torch.FloatTensor(embeddings)
    input = torch.tensor(sentences_aligned)
    if model == 'BOW':
        BOW = BagOfWords( embedding_dim = 50,pretrained_embedding=embeddings_weight)
        # print(sentences_aligned)
        # print(len(sentences_aligned))
        out = BOW(input)
    return out
    #print(word2vec)


def replace_with_dict(sentences,dict_vocab):
    temp_sentences = []
    for sentence in sentences:
        temp_sentence = []
        for word in sentence:
            if word in dict_vocab:
                temp_sentence.append(dict_vocab[word])
            else:
                temp_sentence.append(dict_vocab['#unk#'])
        temp_sentences.append(temp_sentence)
    return temp_sentences


def count_k(words, k):
    dict_counts = {}
    words.sort()
    set_words = set(words)
    list1 = list(set_words)
    list1.sort()
    for i in set_words:
        dict_counts[i] = words.count(i)
    vocabs = [key for key, v in dict_counts.items() if v >= k]
    return vocabs


def eliminate_stop_words(vocabs, stop_words):
    new_vocabs = []
    for word in vocabs:
        if word not in stop_words:
            new_vocabs.append(word)
    return new_vocabs


def align_sentence(sentences, max_len):
    new_sentences = []
    for sentence in sentences:
        if len(sentence) < max_len:
            diff = max_len - len(sentence)
            for i in range(diff):
                sentence.append(1)
                new_sentence = sentence
            new_sentences.append(new_sentence)
        else:
            new_sentence = sentence[:max_len]
            new_sentences.append(new_sentence)
    return new_sentences


class BagOfWords(nn.Module):
    def __init__(self,  embedding_dim,pretrained_embedding):
        super(BagOfWords, self).__init__()
        self.embedding = nn.Embedding.from_pretrained(pretrained_embedding,padding_idx=1)

    def forward(self, x):
        embedded = self.embedding(x)  # (5452, 36, 50)
        print(embedded.size())
        prefix = 1/torch.count_nonzero(torch.count_nonzero(embedded, dim=2), dim=1).reshape(5452,1) # (5452, 1)
        temp_sum = torch.sum(embedded, dim=1)  # (5452, 50)
        output = prefix * temp_sum
        return output



if __name__ == '__main__':
    train_path = 'data/train_5500.label.txt'
    glove_path = 'glove.small/glove.small.txt'
    stop_words_list = ['a', 'and', 'but', 'not', 'up']


    sentences, labels_fine, labels_coarse = load_data(train_path)

    word2vec =create_word_embedding(k=5, all_sentences=sentences, mode='random', stop_words=stop_words_list,model = 'BOW')
    print(word2vec)
    dataset = construct_dataset(word2vec, labels_fine, labels_coarse, mode='coarse')
    train_set, dev_set = split_dataset(dataset, split_coef=0.9)
