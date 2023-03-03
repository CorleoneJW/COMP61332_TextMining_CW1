# README

### **Workflow explanation:**

1. Load data 
    1. whole train data will be split into train data and dev data
2. Create vocab basing on train data
    1. Handle #unk#(1) and #pad#(0)
3. Create word embeddings basing on vocab
    1. Random initialized Embeddings: Create embeddings first, and use randn() to replace #unk#
    2. Glove Embeddings: load glove.txt and prune
4. Create dataset
    1. encode sentence
    2. encode labels
    3. pass train. dev. test set to dataset and dataloader
5. Create model
    1. Model: Bow/BiLSTM: Following .pdf, BiLSTM using last hidden state
    2. Classifier: Dropout, BatchNorm, 1 layer fc and final LogSoftmax
    3. Trainer: Assemble Specific Model & Classifier, also provide train(), val(), test() funcs
6. Parameters
    1. Use argparse and configparse to set all required parameters

### **Command for training:**

Train:

`python3 question_classifier.py --train --config [ config_path]` 

Test:

`python3 question_classifier.py --test --config [ config_path]` 

Example:

`python3 question_classifier.py --train --config 'config_Bow_F.ini'` 

**Consistency**:

In each parameter combination, using `--train` first to train the model, it will output a `trained_model.pth` for `--test` to load Classifier weights.

Please ensure using same config to train, and then test. There is also a `output.txt` which will record the performance on test set. 

Remember the `trained_model.pth` saved model and `output.txt` will be covered when starting a new train/test process under other configs.

### Dataset**:**

Train and dev set are created from `train_5500.label.txt` in advance

Split into `train_set.txt` and `dev_set.txt`

Test set is from `TREC_10.label.txt`

Glove embeddings is from `glove.small.txt`

### Config files**:**

Using windows .ini file as configs.

Here provide, 2 label type: coarse, fine

coarse/fine: each includes 2 models: bow and bilstm

bow/bilstm: each includes 3 settings: random/ pretrained-freeze / pretrained- unfreeze

Example:

For Fine label .ini, F stands for Fine

`config_BiLSTM_F_Pre_Freeze.ini`

`config_BiLSTM_F_Pre.ini`

`config_BiLSTM_F_Random.ini`

`config_Bow_F_Pre_Freeze.ini`

`config_Bow_F_Pre.ini`

`config_Bow_F_Random.ini`

### Requirements:

See `requirements.txt` in Data folder, there is the usage of scikit-learn lib.