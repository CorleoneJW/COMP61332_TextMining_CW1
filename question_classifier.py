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


def load_data(train_path):    #parameter is the path of data
    # data/TREC_10.label.txt
    # data/train_5500.label.txt

    sentences = []    #list of sentence
    labels_fine = []    #list of fine classes
    labels_coarse = []    #list of coarse calsses
    with open(train_path, 'r') as f:    #read model to load
        for line in f:    #ietrate every line of the file
            line_clean = line.strip('\n').split(' ')    #strip the \n and split by the blank space
            sentence = line_clean[1:-1]    #first word is the label, so start with the index 1
            temp_list = []    #temporary list for the words of sentence
            for word in sentence:    #iterate the sentence to get every word
                word = word.lower()    #first treatment for words, becoming lower
                temp_list.append(word)    #append to temp list
            fine = line_clean[0]     #the label, containing the fine and coarse
            coarse = fine.split(':')[0]    #using the syntax : to get coarse
            # fine = label.split(':')[1]
            sentences.append(temp_list)    #append to sentences list
            labels_coarse.append(coarse)    #append to cparse label list
            labels_fine.append(fine)    #append to fine label list
    print(sentences[0:10])    #just check
    print(labels_fine[0:10])
    print(labels_coarse[0:10])
    return sentences, labels_fine, labels_coarse    #return the result


class QuestionDataset(Dataset):    #class of quentionDataset
    def __init__(self, data, labels):    #initial function
        self.data = data
        self.labels = labels

    def __len__(self):    #get the length
        return len(self.data)

    def __getitem__(self, index):    #get the data
        x = self.data[index]
        y = self.labels[index]
        return x, y


def construct_dataset(sentences, labels_fine, labels_coarse, mode):    #according to the model, create different dataset
    if mode == 'coarse':
        dataset = QuestionDataset(sentences, labels_coarse)
        print(dataset)
    else:
        dataset = QuestionDataset(sentences, labels_fine)
        print(dataset)
    #for sentence,label in dataset:

    return dataset


def split_dataset(question_dataset, split_coef):    #split the dataset, parameter are the dataset created before and proportion
    train_size = int(split_coef * len(question_dataset))   #get the size of train dataset
    dev_size = len(question_dataset) - train_size    #get the size of develop dataset
    train_dataset, dev_dataset = random_split(question_dataset, [train_size, dev_size])   #split the dataset
    print(len(train_dataset))
    print(len(dev_dataset))
    return train_dataset, dev_dataset


'''
    Word Embedding

'''


def construct_dict(vocabs_k):    #dictionary for the word, parameter is the k, the least appearance time
    dict_vocabs_k = {'#unk#': 0, '#pad#': 1}   # dictionary for unkown word and pad
    for i, word in enumerate(vocabs_k):    #enumerate, i is the place of word
        dict_vocabs_k[word] = i + 2   #due to already exist #unk# and #pad#, plus 2
        # dict_vocabs_k = {word: i for i, word in enumerate(vocabs_k)}
    return dict_vocabs_k


def create_from_random(vocabs, embedding_size):    #random
    dict_random = construct_dict(vocabs)
    print(dict_random)
    embeddings = nn.Embedding(len(dict_random), embedding_size, padding_idx=1)    #set the embedding of #pad#(1) as 0
    return dict_random,embeddings


def create_from_pretrained(glove_path, vocabs , embedding_size):
    labels, embedding = load_glove(glove_path)     #use the gloVe to get the labels and embeddings
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
    for i in range(len(labels)):    #labels from the gloVe
        if labels[i] in vocabs:   #according the pretrained embedding, add the current one word.
            if labels[i] not in new_dict:    #add the new word
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
    """
       data:
       [('frogs',0.777),('dogs',0.666)]
    """
    result = []
    labels = []
    with open(glove_path, 'r') as f:
        for line in f:
            label = line.split('\t')[0].lower()    #label for the word
            back = line.split('\t')[1]    #the number fo similarity vector?
            line_clean = back.strip('\n').split(' ')
            #result.append(line_clean)
            temp_line = []
            for word in line_clean:
                word_float = float(word)
                temp_line.append(word_float)
            result.append(temp_line)
            labels.append(label)
    return labels,result   #they both corresponding to each other


def create_word_embedding(k, all_sentences, mode, stop_words,model):    #the main funciton of word embedding
    """
       parameters:
       
       k:the least appear time
       all_sentences:all the sentences of the document
       mode:the mode of word embedding, e.g. random
       stop_words:the words appear,stop
       model:model of word embedding, e.g. BOW
    """
    embedding_size = 50    #the size of embedding dimension
    num_iterations = 100    #iteraion times
    all_words = [word for sentence in all_sentences for word in sentence]
    vocabs_k = count_k(all_words, k)   #get the vocabularies for appearing more than k times
    vocabs_k_noStopWords = eliminate_stop_words(vocabs_k, stop_words=stop_words)    #eliminate the stop words
    # sentences = align_sentence(sentences, max_len=5)
    if mode == 'random':    #random mode
        dict_embedding,embeddings = create_from_random(vocabs_k_noStopWords, embedding_size)    #get the random embedding
        embeddings_weight = embeddings.weight    #get the weight of embedding, the category_number*embedding size weight.
    else:    #pretrained embedding, get from the gloVe
        dict_embedding,embeddings = create_from_pretrained(glove_path, vocabs_k_noStopWords,embedding_size=50)
        embeddings_weight = torch.FloatTensor(embeddings)    #only the tensor could be used in torch, convert
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


def replace_with_dict(sentences,dict_vocab):    #replace the dictionary, especially the unkown
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


def count_k(words, k):  #count the words that appear more than k times
    dict_counts = {}    #dictionary for each unique word
    words.sort()    #sort the words
    set_words = set(words)    #get the set of words, using is to strip repeated elements(tokens)
    list1 = list(set_words)    #get a list of set above
    list1.sort()    #sort
    for i in set_words:    #acquire the times of words
        dict_counts[i] = words.count(i)
    vocabs = [key for key, v in dict_counts.items() if v >= k]   #appearing more than k,just get the keys(tokens)
    return vocabs


def eliminate_stop_words(vocabs, stop_words):
    new_vocabs = []
    for word in vocabs:
        if word not in stop_words:   #not in the stopwords, eliminate them
            new_vocabs.append(word)
    return new_vocabs


def align_sentence(sentences, max_len):
    new_sentences = []
    for sentence in sentences:
        if len(sentence) < max_len:
            diff = max_len - len(sentence)
            for i in range(diff):
                sentence.append(1)   #add 1 diff times at the end
                new_sentence = sentence
            new_sentences.append(new_sentence)
        else:
            new_sentence = sentence[:max_len]    #get the max_len part of the sentence
            new_sentences.append(new_sentence)
    return new_sentences


class BagOfWords(nn.Module):   #BOW way
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
    train_path = 'data/train_5500.label.txt'    #load data
    glove_path = 'glove.small/glove.small.txt'    #load glove
    stop_words_list = ['a', 'and', 'but', 'not', 'up']   #stop words


    sentences, labels_fine, labels_coarse = load_data(train_path)

    word2vec =create_word_embedding(k=5, all_sentences=sentences, mode='random', stop_words=stop_words_list,model = 'BOW')
    print(word2vec)
    dataset = construct_dataset(word2vec, labels_fine, labels_coarse, mode='coarse')
    train_set, dev_set = split_dataset(dataset, split_coef=0.9)
