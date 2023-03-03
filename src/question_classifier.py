import random
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
import torch.optim as optim
import argparse
from sklearn.metrics import f1_score
import re
from sklearn.metrics import confusion_matrix
import configparser
import numpy as np

#np.set_printoptions(edgeitems=25)

parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, help='Configuration file',default='config.ini')
parser.add_argument('--train', action='store_true', help='Training mode - model is saved',default=True)
parser.add_argument('--test', action='store_true', help='Testing mode - needs a model to load')


'''

    Set seed, default = 1
    All test is done with seed(1)
    
'''



'''
    ******************************************
    Part 1: loading glove data
        1.1 load glove data, preprocess some issues like lowercase. 
                output: [Labels, [Embeddings]]
        1.2 prune glove data, including create a vocabulary-dict, also, handling duplicated words.
                output: dict, embeddings
'''
def load_glove(glove_path):
    #   Read line by line
    #   handle specific chars and lowercase
    #   Use temp list to store and generating 2 lists
    result = []
    labels = []
    with open(glove_path, 'r') as f:
        for line in f:
            label = line.split('\t')[0].lower()
            back = line.split('\t')[1]
            line_clean = back.strip('\n').split(' ')
            temp_line = []
            for word in line_clean:
                word_float = float(word)
                temp_line.append(word_float)
            result.append(temp_line)
            labels.append(label)

    return labels,result


def prune_glove(labels, embedding,vocabs,embedding_size):
    # Initializing preparation
    vocabs.append('#unk#')
    vocabs.append('#pad#')
    new_labels = []
    new_embedding = []
    new_dict = {}
    t = 2 # start point, since added #unk# and #pad#

    new_dict['#pad#'] = 0
    new_labels.append('#pad#')
    temp_list = []
    for j in range(embedding_size):
        temp_list.append(0)
    new_embedding.append(temp_list)

    new_dict['#unk#'] = 1
    new_labels.append('#unk#')
    for i in range(len(labels)):
        if labels[i] == '#unk#':
            new_embedding.append(embedding[i][:embedding_size])

    for i in range(len(labels)):
        if labels[i] in vocabs:
            if labels[i] not in new_dict:

                new_dict[labels[i]] = t
                new_labels.append(labels[i])
                new_embedding.append(embedding[i][:embedding_size])
                t += 1
    return new_dict,new_embedding


'''
    ******************************************
    Part 2: loading dataset
        2.1 load dataset line by line, also store them separately in lists.
                output: [ [sentences], [coarse labels], [fine labels] ]
        
'''
def load_data(train_path):
    #   data/TREC_10.label.txt
    #   data/train_5500.label.txt
    #   Read line by line
    #   Also handle lowercase
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
            sentences.append(temp_list)
            labels_coarse.append(coarse)
            labels_fine.append(fine)
    return sentences, labels_fine, labels_coarse


def load_presplited_data(file_path):
    #   data/TREC_10.label.txt
    #   data/train_5500.label.txt
    #   Read line by line
    #   Also handle lowercase
    sentences = []
    labels_fine = []
    labels_coarse = []
    with open(file_path, 'r') as f:
        for line in f:
            line_clean = line.strip('\n').split(' ')
            sentence = line_clean[1:]
            temp_list = []
            for word in sentence:
                word = word.lower()
                temp_list.append(word)
            fine = line_clean[0]
            coarse = fine.split(':')[0]
            sentences.append(temp_list)
            labels_coarse.append(coarse)
            labels_fine.append(fine)
    return sentences, labels_fine, labels_coarse
'''
    ******************************************
    Part 3: generating vocab
        3.1 In main(), split train dataset in to 2 parts using zip.
        3.2 create a vocab basing on sentences from training set.
        3.3 keep words that appears K times.
        3.4 Removing words from stop-list. 
        3.5 Clean vocabs with regex, to remove words such as num/letter mixed, or '-' connected word,
                like: '15th' , ' A-B-C'

'''
def create_vocab(train_data,k,stop_words):
    # A entry func to create a final vocab
    all_words = [word for sentence in train_data for word in sentence ]
    vocabs_k = count_k(all_words, k) # get k times appeared
    vocabs_stopWords = remove_stop_words(vocabs_k, stop_words=stop_words) # remove stop words
    vocabs_cleaned = clean_vocabs(vocabs_stopWords) # clean with regex
    return vocabs_cleaned


def count_k(words, k):
    #   Use set to get unique words list.
    #   Sort to make the words list(vocab) clean.
    dict_counts = {}
    words.sort()
    set_words = set(words)
    list1 = list(set_words)
    list1.sort()
    for i in list1:
        dict_counts[i] = words.count(i)
    vocabs = [key for key, v in dict_counts.items() if v >= k]
    return vocabs


def remove_stop_words(vocabs, stop_words):
    #   One by one check if exists in the stop words list.
    vocabs_new = []
    for word in vocabs:
        if word not in stop_words:
            vocabs_new.append(word)
    return vocabs_new


def clean_vocabs(words):
    #   Use regex to check each word in the vocab
    #   Some are removed in STOP_WORDS previously
    #   Remove especially nums, nums+letters mixed, '-' connected words
    pattern = r"[a-z]+"
    matches = []
    for word in words:
        if re.findall(pattern,word):
            matches.append(word)
    pattern1 = '[0-9]'
    pattern2 = '\-'
    matches1 = []
    for word in matches:
        if re.findall(pattern1, word): # find excluding num
            pass
        else:
            if re.findall(pattern2, word): # find excluding '-'
                pass
            else:
                matches1.append(word)
    return matches1


'''
    ******************************************
    Part 4: generating word embedding
        2 ways, using random or pretrained.
        4.1 a entry func to create word embedding
        4.2 Random initializing embeddings, which exactly using nn.Embedding to create first
                and pass it to model (bow,bilstm) later 
                using from_pretrained() to keep it same with using pretrained model
        4.3 pretrained embeddings, which is basing on Glove embeddings in data.
                using predefined load and prune func to read and use.
        output: dict(which is a little bit different from vocabs), embedding_weight (weight tensor,not embedding object) 
        

'''
def create_word_embedding(vocabs, embedding_mode,embedding_size,glove_path):
    #   random-part
    if embedding_mode == 'random':
        dict_emb,embeddings = create_from_random(vocabs, embedding_size)
        embeddings_weight = embeddings.weight
        embeddings_weight.requires_grad = False
        unk_embedding = torch.randn(1, embedding_size)

        embeddings_weight[1] = unk_embedding
    #   pretrained-part
    elif embedding_mode == 'pretrained':
        dict_emb,embeddings = create_from_pretrained(glove_path, vocabs, embedding_size=embedding_size)
        embeddings_weight = torch.FloatTensor(embeddings)
    return dict_emb,embeddings_weight


def create_dict_random(vocabs):
    #   a dic that predefined, serve for latter encoding sentence
    dict_vocabs = {'#pad#': 0, '#unk#': 1}
    for i, word in enumerate(vocabs):
        dict_vocabs[word] = i + 2
    return dict_vocabs


def create_from_random(vocabs, embedding_size):
    #   Random embedding, size [ dict length, embedding size ]
    #   dict length includes #pad# #unk#
    dict_random = create_dict_random(vocabs)
    embeddings = nn.Embedding(len(dict_random), embedding_size, padding_idx=0)
    return dict_random,embeddings


def create_from_pretrained(glove_path, vocabs , embedding_size):
    #   Pretrained embedding, size [ dict length, embedding size ]
    #   dict length includes #pad# #unk#
    #   not using nn.embedding, load from glove.
    labels, embedding = load_glove(glove_path)
    dict_glove,embeddings_glove =prune_glove(labels, embedding,vocabs,embedding_size)
    return dict_glove,embeddings_glove


'''
    ******************************************
    Part 5: Model BOW and BiLSTM
        5.1 a entry func to generate a model and return
            also handle the option for freeze or not.
        5.2 BOW
            using from_pretrained to load embeddings (both random and pretrained)
            using find_length to find the actual length of each sentence, reshape [ batch size, 1 ]
            output [ batch_size, embedding_dim ]
        5.3 BiLSTM
            using from_pretrained to load embeddings (both random and pretrained)
            using the last hidden state from both forward and backward, then concat.
            output [ batch_size, hidden_dim *2 ]

'''
def produce_model(embeddings_weight,model_name,freeze_pretrained,embedding_size,hidden_dim):
    #   Entry func
    #   First set freeze or not since it will pass through model.
    #   Generate model and return, using mode_name to control.
    if freeze_pretrained:
        embeddings_weight.requires_grad = False
    else:
        embeddings_weight.requires_grad = True
    if model_name == 'bow':
        BOW = BagOfWords( pretrained_embedding=embeddings_weight,freeze_pretrained=freeze_pretrained)
        return BOW
    elif model_name == 'bilstm':
        BILSTM = BiLSTM(embedding_dim=embedding_size, hidden_dim = hidden_dim,pretrained_embedding=embeddings_weight,freeze_pretrained=freeze_pretrained)
        return BILSTM


def find_sum_sentence(sentences):
    temp_sentence_emb = []
    sum_tensor = torch.zeros(size = (sentences.size()))
    #print(sum_tensor.size())
    for i in range(sentences.size()[0]):
        for j in range(sentences.size()[1]):
            if sentences[i][j].item() == 1:
                sentences[i][j] = torch.zeros((1,1))
    return sentences



class BagOfWords(nn.Module):
    def __init__(self,  pretrained_embedding,freeze_pretrained):
        super(BagOfWords, self).__init__()
        self.embeddings = nn.Embedding.from_pretrained(pretrained_embedding,freeze=freeze_pretrained,padding_idx=0)

    def forward(self, x):
        temp = torch.zeros(size=(x.size()[0],self.embeddings.weight.size()[1]))
        x = find_sum_sentence(x)

        #   y for non-zero, means length including #unk#, also actual length.
        #   z for non-zero, non-one, means length excluding #unk#.
        #       z has risk of 0, so add a minimum 1.

        y = torch.count_nonzero(x, dim=1)
        embedded = self.embeddings(x)# [ batch size, max sentence length, embedding dim ]
        prefix = y.reshape(x.size()[0], 1)
        #output = torch.sum(embedded, dim=1)/prefix # [ batch size, embedding dim ]
        temp_sum = torch.sum(embedded, dim=1)
        for i in range(prefix.size()[0]):
            if prefix[i].item() == 0:
                pass
            else:
                temp[i] = temp_sum[i]/prefix[i]
        #output = prefix * temp_sum
        return temp # [ batch size, embedding dim ]


class BiLSTM(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, pretrained_embedding,freeze_pretrained):
        super(BiLSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.embeddings = nn.Embedding.from_pretrained(pretrained_embedding,freeze=freeze_pretrained,padding_idx=0)
        # batch_first to match the input format
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, bidirectional=True,batch_first=True)

    def forward(self, sentence):

        embeds = self.embeddings(sentence) # [ batch size, max sentence length, embedding dim ]
        lstm_out, (h_n,_) = self.lstm(embeds) # considering using last hidden state, [ batch size, 2 , hidden dim *2]
        final_hidden_state = torch.cat((h_n[0, :, :], h_n[1, :, :]), dim=1) # [ batch size,  hidden dim *2]
        return final_hidden_state


'''
    ******************************************
    Part 6: dataset preparation
        6.1 Define a dataset class to help create dataset and pass it to data loader.

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


'''
    ******************************************
    Part 7: dataset preparation
        7.1 Entry func
        7.2 label preprocess, encoding label from 0 to 5(49)
        7.3 enconde sentence with dict, also align sentence to a predefined length

'''
def data_preprocess(sentence,label_f,label_c,dict_vocabs,max_len,label_mode,dict_labels_F,dict_labels_C):
    #   Entry func
    #   process both coarse and fine labels, coarse -> C, fine -> F
    labels_F_encoded = label_preprocess(label_f,dict_labels_F)
    labels_C_encoded = label_preprocess(label_c,dict_labels_C)
    sentences_encoded = encode_sentence(sentence, dict_vocabs)
    sentences_aligned = align_sentence(sentences_encoded, max_len=max_len)
    input = torch.tensor(sentences_aligned)
    if label_mode == 'coarse':
        labels = labels_C_encoded
    elif label_mode =='fine':
        labels = labels_F_encoded
    return input,labels


def create_label_dict(all_labels):
    labels_set = set(all_labels)
    list1 = list(labels_set)
    list1.sort()
    dict_labels = {}
    ind = 0
    for i in list1:
        dict_labels[i] = ind
        ind += 1
    return dict_labels


def label_preprocess(labels,dict_labels):
    #   Using set to get unique label and sort to get a dict
    #   encode labels with dict values (num)
    labels_set = set(labels)
    list1 = list(labels_set)
    list1.sort()
    new_list = []
    for i in labels:
        new_list.append(dict_labels[i])

    return new_list


def encode_sentence(sentences,dict_vocab):
    #   Encode sentences with dict
    #   If a word not in dict, use 0 (#unk#) to replace.
    temp_sentences = []
    for sentence in sentences:
        temp_sentence = []
        for word in sentence:
            if word in dict_vocab:
                temp_sentence.append(dict_vocab[word]) # including #pad# which is 1
            else:
                temp_sentence.append(dict_vocab['#unk#']) # 0
        temp_sentences.append(temp_sentence)
    return temp_sentences


def align_sentence(sentences, max_len):
    #   If len < max_len, use 0 to pad
    #   Else, cut [ 0: max_len]
    new_sentences = []
    for sentence in sentences:
        if len(sentence) < max_len:
            diff = max_len - len(sentence)
            for i in range(diff):
                sentence.append(0)
                new_sentence = sentence
            new_sentences.append(new_sentence)
        else:
            new_sentence = sentence[:max_len]
            new_sentences.append(new_sentence)
    return new_sentences


'''
    ******************************************
    Part 8: Classifier 
        A simple forward feed classifier, last layer using Logsoftmax
        Input -> BN,Dropout [ batch_size * input_dim ]
                            [ input_dim basing on bilstm or bow] 
        1 layer fc -> output -> logsoftmax  [ batch_size, output_dim]
                                            [ output_dim basing on label type 6 or 50]
'''
class QuestionClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(QuestionClassifier, self).__init__()
        self.fc1 = nn.Linear(input_dim, output_dim)
        self.bn1 = nn.BatchNorm1d(input_dim)
        self.dropout1 = nn.Dropout(0.3)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        #self.fc3 = nn.Linear(hidden_dim, output_dim)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        x = self.dropout1(x)
        x = self.bn1(x)
        #x = self.dropout1(x)
        x = self.fc1(x)
        #x = self.fc2(x)
        x = self.softmax(x)

        return x


'''
    ******************************************
    Part 9: Trainer
        a Trainer contains both BOW/BiLSTM and Classifier.
        Also provide train(), validate(), test() func to check on train set and dev set
        train: 
            After each epoch, output accuracy of train set
            Then switch to validate() (also called dev)
        record the best f1-score, record best classifier weights, produce a .pth file to store model weights.
        dev:
            Print accuracy, F1-score, confusion matrix   
        test:
            similar to dev, but need to write output to a .txt file.            
            
'''

class ClassifierTrainer(object):

    def __init__(self, model, classifier, train_loader, test_loader, dev_loader, optimizer, loss_fn, \
                 model_name, embedding_mode, label_mode,freeze_embedding):
        self.optimizer = optimizer
        self.model = model
        self.classifier = classifier
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.dev_loader = dev_loader
        self.loss_fn = loss_fn
        self.model_name = model_name
        self.embedding_mode = embedding_mode
        self.label_mode = label_mode
        self.freeze_embedding = freeze_embedding

    def train(self, n_epochs):
        self.classifier.train()

        best_f1 = 0
        best_cm = 0
        best_cm_label = 0
        for epoch in range(n_epochs):
            #print(self.model.embeddings.weight)
            temp_acc = 0
            temp_result = 0
            for inputs, labels in self.train_loader:
                sentence_repr = self.model(inputs)
                self.optimizer.zero_grad()
                outputs = self.classifier(sentence_repr)
                loss = self.loss_fn(outputs, labels)
                loss.backward()
                self.optimizer.step()
                _, pred= torch.max(outputs.data, 1)
                results = pred == labels
                correct_points = torch.sum(results.long())
                temp_acc += correct_points.float()
                temp_result += results.size()[0]
            if (epoch + 1) % 1 == 0:
                print('Performance with epoch', epoch+1)
                acc= (temp_acc/temp_result *100).item()
                print('acc-train: ', acc)
                best_f1,best_cm,best_cm_label = self.validate(epoch,best_f1,best_cm,best_cm_label)
        print('Best F1-score: ',best_f1)
        print('Best CM: ')
        print(best_cm)
        #print(best_cm_label)
        np.savetxt('output.csv', best_cm, delimiter=', ')


    def validate(self,epoch,best_f1,best_cm,best_cm_label):
        self.classifier.eval()
        with torch.no_grad():
            temp_label = []
            temp_pred = []
            temp_acc = 0
            temp_result = 0
            for inputs, labels in self.dev_loader:
                sentence_repr = self.model(inputs)
                outputs = self.classifier(sentence_repr)
                loss = self.loss_fn(outputs, labels)
                _, pred = torch.max(outputs.data, 1)
                for i in labels:
                    temp_label.append(i.item())
                for j in  pred:
                    temp_pred.append(j.item())
                results = pred == labels
                correct_points = torch.sum(results.long())
                temp_acc += correct_points.float()
                temp_result += results.size()[0]
            acc = (temp_acc / temp_result * 100).item()
            print('acc-dev: ', acc)
            f1 = f1_score(temp_label, temp_pred,average='macro') * 100
            cm = confusion_matrix(temp_label, temp_pred)

            print('F1-Score: ', f1)
            print('Confusion Matrix:')
            print(cm)
            label = (set(temp_label+temp_pred))
            if epoch == 0:
                best_weights = self.classifier.state_dict()
                torch.save(self.classifier.state_dict(), 'trained_model.pth')
                best_f1 = f1
                best_cm = cm
                best_cm_label = label
            else:
                if f1 >= best_f1:
                    best_f1 = f1
                    best_cm = cm
                    best_cm_label = label
                    best_weights = self.classifier.state_dict()
                    torch.save(self.classifier.state_dict(), 'trained_model.pth')
            return best_f1,best_cm,best_cm_label



    def test(self,output_path):
        self.classifier.eval()
        with torch.no_grad():
            temp_label = []
            temp_pred = []
            temp_acc = 0
            temp_result = 0
            for inputs, labels in self.test_loader:
                sentence_repr = self.model(inputs)
                outputs = self.classifier(sentence_repr)
                #loss = self.loss_fn(outputs, labels)
                _, pred = torch.max(outputs.data, 1)
                for i in labels:
                    temp_label.append(i.item())
                for j in  pred:
                    temp_pred.append(j.item())
                results = pred == labels
                correct_points = torch.sum(results.long())
                temp_acc += correct_points.float()
                temp_result += results.size()[0]
            acc = (temp_acc / temp_result * 100).item()
            print('acc-test: ', acc)
            f1 = f1_score(temp_label, temp_pred,average='macro') * 100
            cm = confusion_matrix(temp_label, temp_pred)
            with open(output_path, "w") as output_file:
                #output_file.write()
                for i in range(len(temp_label)):
                    str1 = str('Original Label: '+str(temp_label[i])+', \t' +' Prediction: '+ str(temp_pred[i])+'\n')
                    output_file.write(str1)
                output_file.write(str('Final acc for test set: '+ str(acc)+'\n'))
                output_file.write(str('Final F1-score for test set: ' + str(f1)))
            print('F1-Score: ',f1)
            print('Confusion Matrix:')
            print(cm)




if __name__ == '__main__':
    '''
        Initializing setting
        arg, configarg
    '''
    args = parser.parse_args()
    config = configparser.ConfigParser()
    config.sections()
    config.read(args.config)
    '''
        Load parameters from config file.
    '''
    random_seed = int(config['hyperparameters']['random_seed'])
    trainset_path = config['data_path']["train_path"]
    devset_path = config['data_path']["dev_path"]
    glove_path = config['data_path']["glove_path"]
    testset_path = config['data_path']["test_path"]
    model_name = config['model_params']['model_name'] # 'bow' 'bilstm', lowercase only
    label_mode =  config['model_params']['label_mode']  # 'coarse' 'fine' , lowercase only
    embedding_mode = config['model_params']['embedding_mode']  # 'pretrained' 'random' , lowercase only
    freeze_pretrained = config['model_params']['freeze_pretrained'] # True False
    if freeze_pretrained =='True':
        freeze_pretrained = True
    elif freeze_pretrained =='False':
        freeze_pretrained = False
    embedding_dim = int(config['sentence_params']['embedding_dim']) # 300 , could ba larger for random embeddings
    max_len = int(config['sentence_params']['max_len']) # 36, can be larger, smaller
    k = int(config['sentence_params']['k']) # 5, k- at least appeared
    split_coef = 0.9 # split train set to 90% : 10%
    batch_size = int(config['hyperparameters']['batch_size']) # 300
    learning_rate = float(config['hyperparameters']['learning_rate'] ) # 0.007
    hidden_dim = int(config['hyperparameters']['hidden_dim']) # 300, for bilstm hidden dim
    input_dim = 2 * hidden_dim if model_name == 'bilstm' else embedding_dim
    hidden_dim2 = int(config['hyperparameters']['hidden_dim2']) # 800, for classifier hidden dim, may be not used
    n_classes = 6 if label_mode == 'coarse' else 50
    n_epoch = int(config['hyperparameters']['epochs']) # 10
    output_path = config['data_path']['output_path']
    stop_words = ['a', 'and', 'but', 'not', 'up', '!', '.', '$1', '$5', '&', "'", "''",
                "'clock", "'em", "'hara", "'l", "'ll", "'n", "'re", "'s"
                    , "'t", "'ve", ",", '-', '?', ':', '``', '`']
    ''''''
    '''
    Set seed
    '''
    torch.manual_seed(random_seed)
    random.seed(random_seed)

    # load splitted train,dev,test set
    train_sentences, train_labels_F, train_labels_C = load_data(trainset_path) # train_set path
    dev_sentences, dev_labels_F, dev_labels_C = load_data(devset_path)
    test_sentences, test_labels_F, test_labels_C = load_data(testset_path)
    #data = list(zip(train_sentences, train_labels_F,train_labels_C))

    # label dict
    dict_labels_F = create_label_dict(train_labels_F)
    dict_labels_C = create_label_dict(train_labels_C)
    print(dict_labels_F)
    print(dict_labels_C)

    ### For first time create dataset, load train_5500.label.txt
    ### zip data and shuffle and then split
    #
    # data = list(zip(train_sentences, train_labels_F,train_labels_C))
    # shuffle and split train, dev data from train_all data
    # random.shuffle(data)
    # split_index = int(len(train_sentences) * split_coef)
    # train_data, train_labels_f,train_labels_c = zip(*data[:split_index])
    # dev_data, dev_labels_f,dev_labels_c = zip(*data[split_index:])

    vocabs = create_vocab(train_sentences, k, stop_words)
    #print(vocabs)
    # func for first time write train,dev set to files
    #

    # with open('train_set.txt', "w") as f:
    #     # output_file.write()
    #     for i in range(len(train_labels_f)):
    #         str1 = ''
    #         str1 += str(train_labels_f[i])
    #         str1 += ' '
    #         for j in range(len(train_data[i])):
    #             str1 += str(train_data[i][j])+' '
    #         str1 += '\n'
    #         f.write(str1)
    # with open('dev_set.txt', "w") as f:
    #     # output_file.write()
    #     for i in range(len(dev_labels_f)):
    #         str1 = ''
    #         str1 += str(dev_labels_f[i])
    #         str1 += ' '
    #         for j in range(len(dev_data[i])):
    #             str1 += str(dev_data[i][j]) + ' '
    #         str1 += '\n'
    #         f.write(str1)

    # vocab dict, embeddings.weight
    dict_vocabs, embeddings = create_word_embedding(embedding_mode=embedding_mode,vocabs=vocabs,
                                                    embedding_size=embedding_dim,glove_path=glove_path
                                                    )
    # encode label, sentence basing on dict vocab and dict label
    train_input,train_label = data_preprocess(sentence=train_sentences,label_c=train_labels_C,label_f=train_labels_F,
                                              max_len=max_len,label_mode=label_mode,dict_vocabs=dict_vocabs,
                                              dict_labels_C=dict_labels_C,dict_labels_F=dict_labels_F)
    dev_input, dev_label = data_preprocess(sentence=dev_sentences, label_c=dev_labels_C, label_f=dev_labels_F,
                                               max_len=max_len, label_mode=label_mode,dict_vocabs=dict_vocabs,
                                           dict_labels_C=dict_labels_C,dict_labels_F=dict_labels_F)
    test_input, test_label = data_preprocess(sentence=test_sentences, label_c=test_labels_C, label_f=test_labels_F,
                                           max_len=max_len, label_mode=label_mode, dict_vocabs=dict_vocabs,
                                             dict_labels_C=dict_labels_C,dict_labels_F=dict_labels_F)

    # create dataset, and dataloader for train,dev,test
    train_dataset = QuestionDataset(train_input, train_label)
    trainset_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    dev_dataset = QuestionDataset(dev_input, dev_label)
    devset_loader = torch.utils.data.DataLoader(dev_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    test_dataset = QuestionDataset(test_input, test_label)
    testset_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=4)


    # create BOW or BiLSTM
    # create Optimizer
    # create Trainer
    model = produce_model(embeddings, model_name=model_name, freeze_pretrained=freeze_pretrained,
                          embedding_size=embedding_dim, hidden_dim=hidden_dim)
    # optimizer = optim.SGD(model.parameters(),lr=1e-3, weight_decay=args.weight_decay, momentum=0.9)
    Classifier = QuestionClassifier(input_dim=input_dim, hidden_dim=hidden_dim2, output_dim=n_classes)
    optimizer = optim.Adam([{'params': model.parameters()},
                            {'params': Classifier.parameters()}], lr=learning_rate,weight_decay=0.001)

    Trainer = ClassifierTrainer(model=model, classifier=Classifier, train_loader=trainset_loader,
                                dev_loader=devset_loader, test_loader=testset_loader,
                                optimizer=optimizer, loss_fn=nn.CrossEntropyLoss(), model_name=model_name,
                                embedding_mode=embedding_mode
                                , label_mode=label_mode, freeze_embedding=freeze_pretrained)

    if args.train:
        # call train function
        Trainer.train(n_epoch)
    elif args.test:
        # call test function
        Classifier.load_state_dict(torch.load('trained_model.pth'))
        Trainer.test(output_path)
    #Trainer.test(output_path)