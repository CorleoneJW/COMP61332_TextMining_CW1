import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
import torch.optim as optim
import argparse

parser = argparse.ArgumentParser()



torch.manual_seed(1)
random.seed(1)


'''

    Part1:
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


'''
    Part2:
    Data set
    
'''
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


def construct_dataset(sentences, labels_fine, labels_coarse, label_mode):
    if label_mode == 'coarse':
        dataset = QuestionDataset(sentences, labels_coarse)
    elif label_mode == 'fine':
        dataset = QuestionDataset(sentences, labels_fine)
    return dataset


def split_dataset(question_dataset, split_coef):
    train_size = int(split_coef * len(question_dataset))
    dev_size = len(question_dataset) - train_size
    train_dataset, dev_dataset = random_split(question_dataset, [train_size, dev_size])
    print(len(train_dataset))
    print(len(dev_dataset))
    return train_dataset, dev_dataset


'''

    Part 3:
    glove data


'''
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

'''

    Part4:
    Word Embedding:


'''

def create_word_embedding(k, all_sentences, embedding_mode, stop_words,embedding_size):

    all_words = [word for sentence in all_sentences for word in sentence]
    print(all_words[:10])
    vocabs_k = count_k(all_words, k)
    print(vocabs_k[:10])
    vocabs_k_noStopWords = eliminate_stop_words(vocabs_k, stop_words=stop_words)
    print(vocabs_k_noStopWords)
    # sentences = align_sentence(sentences, max_len=5)
    if embedding_mode == 'random':
        dict_emb,embeddings = create_from_random(vocabs_k_noStopWords, embedding_size)
        embeddings_weight = embeddings.weight
    elif embedding_mode == 'pretrained':
        dict_emb,embeddings = create_from_pretrained(glove_path, vocabs_k_noStopWords,embedding_size=embedding_size)
        embeddings_weight = torch.FloatTensor(embeddings)

    #model = produce_model(dict_embedding, embeddings_weight, all_sentences,max_len = 40,model = model)
    #embeddings_weight.requires_grad = freeze_pretrained
    #print(embeddings_weight)
    return dict_emb,embeddings_weight


def produce_model(embeddings_weight,model_name,freeze_pretrained,embedding_size):
    #sentences_replaced = replace_with_dict(sentences,dict_vocab)
    #sentences_aligned = align_sentence(sentences_replaced, max_len)
    #weight = torch.FloatTensor(embeddings)
    input = torch.tensor(sentences_aligned)
    if freeze_pretrained:
        embeddings_weight.requires_grad = False
    else:
        embeddings_weight.requires_grad = True
    if model_name == 'bow':
        BOW = BagOfWords( embedding_dim = embedding_size,pretrained_embedding=embeddings_weight)
        # print(sentences_aligned)
        # print(len(sentences_aligned))
        #out = BOW(input)
        return BOW
    elif model_name == 'bilstm':
        BILSTM = BiLSTM(embedding_dim=50, hidden_dim = 15,pretrained_embedding=embeddings_weight)
        #out = BILSTM(input)
        return BILSTM
    #return out
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
    for i in list1:
        dict_counts[i] = words.count(i)
    vocabs = [key for key, v in dict_counts.items() if v >= k]
    return vocabs

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
    #print(labels)
    len_labels = len(labels) #9549
    dict_glove,embeddings_glove =prune_glove(labels, embedding,vocabs,embedding_size)
    return dict_glove,embeddings_glove

'''
    Part 5:
    Preprocess with input data

'''
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



'''
    Part6
    Bag-of-words & BiLstm

'''


class BagOfWords(nn.Module):
    def __init__(self, embedding_dim, pretrained_embedding):
        super(BagOfWords, self).__init__()
        self.embedding = nn.Embedding.from_pretrained(pretrained_embedding, padding_idx=1)

    def forward(self, x):
        embedded = self.embedding(x)  # (5452, 36, 50)
        prefix = 1 / torch.count_nonzero(torch.count_nonzero(embedded, dim=2), dim=1).reshape(embedded.size()[0], 1)  # (5452, 1)
        temp_sum = torch.sum(embedded, dim=1)  # (5452, 50)
        output = prefix * temp_sum

        return output


class BiLSTM(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, pretrained_embedding):
        super(BiLSTM, self).__init__()
        self.hidden_dim = hidden_dim
        # self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.word_embeddings = nn.Embedding.from_pretrained(pretrained_embedding, padding_idx=1)
        # The BiLSTM takes word embeddings as inputs, and outputs hidden states
        # with dimensionality hidden_dim.
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, bidirectional=True)
        # The linear layer that maps from hidden state space to tag space
        # self.hidden2tag = nn.Linear(hidden_dim, tagset_size)

    def forward(self, sentence):
        embeds = self.word_embeddings(sentence)


        # lstm_out, _ = self.lstm(embeds.view(len(sentence), 1, -1))
        lstm_out, _ = self.lstm(embeds)
        lstm_out = lstm_out.mean(dim=1)
        # lstm_out = lstm_out[-1]
        #print(lstm_out.size())
        # tag_space = self.hidden2tag(lstm_out.view(len(sentence), -1))
        # tag_scores = nn.functional.log_softmax(tag_space, dim=1)
        return lstm_out

'''
    Part7
    Classifier

'''
class QuestionClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(QuestionClassifier, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, sentence):
        x = self.fc1(sentence)
        #x = nn.functional.relu(x)
        x = self.fc2(x)
        #print(x.size())
        x = self.softmax(x)
        #print(x.size())
        return x


'''
    Part8
    Trainer
'''

class ClassifierTrainer(object):

    def __init__(self, model, classifier, train_loader, val_loader, optimizer, loss_fn, \
                 model_name, embedding_mode, label_mode,freeze_embedding):
        self.optimizer = optimizer
        self.model = model
        self.classifier = classifier
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.loss_fn = loss_fn
        self.model_name = model_name
        self.embedding_mode = embedding_mode
        self.label_mode = label_mode
        self.freeze_embedding = freeze_embedding

    def train(self, n_epochs):
        self.classifier.train()
        for epoch in range(n_epochs):
            #print(self.model.embedding.weight)
            for inputs, labels in self.train_loader:

                sentence_repr = self.model(inputs)

                self.optimizer.zero_grad()
                # Forward pass through the feed-forward neural network model
                outputs = self.classifier(sentence_repr)
                #print(input[0])

                #print(outputs[0])
                # Compute the loss and perform backpropagation


                loss = self.loss_fn(outputs, labels)
                #print('*start* ' *10)
                #print(self.classifier.fc1.weight.grad)

                loss.backward()
                print(self.classifier.fc1.weight.grad)
                #print(self.classifier.fc1.weight)
                self.optimizer.step()
                #print(self.classifier.fc1.weight)
                _, pred= torch.max(outputs.data, 1)

                results = pred == labels

                correct_points = torch.sum(results.long())

                acc = correct_points.float() / results.size()[0]

            if (epoch + 1) % 1 == 0:
                print('Hidden layer weights at epoch', epoch)
                print(self.classifier.fc1.weight)

    def validate(self):
        self.model.eval()


def label_preprocess(labels):
    labels_set = set(labels)
    print(labels_set)
    list1 = list(labels_set)
    list1.sort()
    dict_labels ={}
    ind = 0
    for i in list1:
        dict_labels[i] = ind
        ind += 1
    print(dict_labels)
    new_list = []
    for i in labels:
        new_list.append(dict_labels[i])
    return new_list

if __name__ == '__main__':


    args = parser.parse_args()

    ''''''
    k = 5
    train_path = 'data/train_5500.label.txt'
    test_path = 'data/TREC_10.label.txt'
    glove_path = 'glove.small/glove.small.txt'
    stop_words_list = ['a', 'and', 'but', 'not', 'up']
    model_name = 'bow' # 'bow'
    label_mode = 'fine' # 'coarse'
    embedding_mode = 'pretrained' # 'pretrained'
    freeze_pretrained = False # True
    embedding_dim = 200
    max_len = 30
    split_coef = 0.9
    batch_size = 20
    learning_rate = 0.15
    ''''''

    sentences, labels_fine, labels_coarse = load_data(train_path)
    dict_vocab, embeddings = create_word_embedding(k=k, all_sentences=sentences, embedding_mode=embedding_mode,
                                                   stop_words=stop_words_list,embedding_size=embedding_dim
                                                   )
    label_fine_replaced = label_preprocess(labels_fine)
    label_coarse_replaced = label_preprocess(labels_coarse)

    sentences_replaced = replace_with_dict(sentences, dict_vocab)
    sentences_aligned = align_sentence(sentences_replaced, max_len=max_len)
    input = torch.tensor(sentences_aligned)
    dataset = construct_dataset(input, label_fine_replaced, label_coarse_replaced, label_mode=label_mode)
    train_set, dev_set = split_dataset(dataset, split_coef=split_coef)
    trainset_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=4)

    model = produce_model(embeddings, model_name=model_name,freeze_pretrained=False,embedding_size=embedding_dim)
    #input = torch.tensor(sentences_aligned)


    #optimizer = optim.SGD(model.parameters(),lr=1e-3, weight_decay=args.weight_decay, momentum=0.9)
    Classifier = QuestionClassifier(input_dim=200, hidden_dim=100, output_dim=50)
    optimizer = optim.SGD(Classifier.parameters(), lr=learning_rate)


    Trainer = ClassifierTrainer(model=model, classifier=Classifier, train_loader=trainset_loader, val_loader=0,
                              optimizer=optimizer, loss_fn=nn.CrossEntropyLoss(), model_name=model_name, embedding_mode= embedding_mode
                                                ,label_mode = label_mode, freeze_embedding=freeze_pretrained)
    Trainer.train(5)
