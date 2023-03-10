# Description of funcs

### Part 1: Glove embeddings preparation

load_glove(path): load glove.small.txt, line by line, return label/item and corresponding embs in 2 lists.

prune_glove(embs, vocab…): Basing on the vocabulary to select required glove embeddings.

### Part 2: Load data

load_data(path): load train,dev,test data line by line, can also be used to load whole data set from train_5500.label.txt. then split whole dataset into 2 parts in the main func.

### Part 3: Generate vocab

create_vocab(): Entry func

count_k(vocab): select words that appeard at least k times

remove_stop_words(vocab): remove stop words from vocab list

clean_vocabs(vocab): use regex to remove some special words.

### Part 4: Generate word embeddings

create_word_embedding(): Entry func

create_dict_random(vocab): create a dictionary for ‘random’ embeddings, #unk#, #pad# and vocab

create_from_random(): call previous func to generate dict and embeddings

create_from_pretrained(): call Part1. funcs to generate dict and embeddings

### Part 5: Modelling BOW and Bilstm

produce_model(): entry func

find_sum_sentence(sentences): supporting func for BOW, it casts #unk# which is 1, to #pad# token, which is 0. This is helpful to sum the vector and count the length of sentence.

BagOfWords(): BOW class, accept sentence first, use previous func to cast #unk# here to #pad#. Then calculate length and sum of involved embeddings, finally output sentence repr

BiLSTM(): BiLSTM class, accept sentence first,  then call word embeddings, then pass to lstm. 

Use hidden state as output, concat the forward h_n and backward h_n. Finally output sentence repr

### Part 6: Dataset preparation

QuestionDataset(): dataset class

data_preprocess(): Entry func

create_label_dict(): get unique dict of labels

label_preprocess(): encode label

encode_sentences(): encode sentence

align sentence(): padding, use 0 as #pad#

### Part 7: Classifier

QuestionClassifier(): class, have dropout, bn, fc and logsoftmax layer

### Part 8: Trainer

ClassifierTrainer(): class, provide train(),validate(),test(), produce acc, f1-score, confusion_matrix

In the end of each epoch in train(), run validate()

In the end of train(). save the best model weight to ‘trained_model.pth’

Also, print best F1-score and confusion matrix

Test() is similar to validate(). but write a ‘output.txt’ to record prediction, original labels, acc and f1-score.

### Part 9: Main()

Including set parameters from argparse and configparse.

Create vocab, word embeddings, models(BOW/Bilstm)

Preprocess sentence, label with encoding.

Create instance of Classifier, Optimizer, Trainer

Call Trainer.train() to train and validate.

Call Trainer.test() to test.