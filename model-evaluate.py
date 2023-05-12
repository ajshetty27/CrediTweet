import numpy as np
import pandas as pd
import transformers
from transformers import AutoModel, BertTokenizerFast
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from transformers import AdamW

#Load in dataset
df=pd.read_csv('main_dataset_v2.csv')

df.head()

#Train-test split
train_text, temp_text, train_labels, temp_labels = train_test_split(df['title'], df['label'],random_state=2018,test_size=0.3,stratify=df['tag'])
# Validation-Test split
val_text, test_text, val_labels, test_labels = train_test_split(temp_text, temp_labels, random_state=2018, test_size=0.5, stratify=temp_labels)

#Load in Bert model
bert = AutoModel.from_pretrained('bert-base-uncased')
#Load in bert tokenizer
tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')

MAX_LENGHT = 15
# Tokenize and encode sequences in the training set
tokens_train = tokenizer.batch_encode_plus(train_text.tolist(),max_length = MAX_LENGHT,pad_to_max_length=True,truncation=True)
# Tokenize and encode sequences in the validation set
tokens_val = tokenizer.batch_encode_plus(val_text.tolist(),max_length = MAX_LENGHT,pad_to_max_length=True,truncation=True)
# tokenize and encode sequences in the test set
tokens_test = tokenizer.batch_encode_plus(test_text.tolist(),max_length = MAX_LENGHT,pad_to_max_length=True,truncation=True)

#Define train seq, mask and label
train_seq = torch.tensor(tokens_train['input_ids'])
train_mask = torch.tensor(tokens_train['attention_mask'])
train_y = torch.tensor(train_labels.tolist())

#Define validation sequence, mask and label
val_seq = torch.tensor(tokens_val['input_ids'])
val_mask = torch.tensor(tokens_val['attention_mask'])
val_y = torch.tensor(val_labels.tolist())

#Define testing sequence, mask and label
test_seq = torch.tensor(tokens_test['input_ids'])
test_mask = torch.tensor(tokens_test['attention_mask'])
test_y = torch.tensor(test_labels.tolist())

#Define Batch size for training
batch_size = 64
# wrap tensors
train_data = TensorDataset(train_seq, train_mask, train_y)
#Create sampler for sampling the data during training
train_sampler = RandomSampler(train_data)
#Load data for train set
train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)
#Wrap tensors
val_data = TensorDataset(val_seq, val_mask, val_y)
#Create sampler for sampling the data during training
val_sampler = SequentialSampler(val_data)
#Load data for validation set
val_dataloader = DataLoader(val_data, sampler = val_sampler, batch_size=batch_size)

# Freezing the parameters and defining trainable BERT structure
for param in bert.parameters():
    #False here means gradient need not be computed
    param.requires_grad = False

class BERT_Arch(nn.Module):
    def __init__(self, bert):
      super(BERT_Arch, self).__init__()
      self.bert = bert
      #dropout layer
      self.dropout = nn.Dropout(0.1)
      #relu activation function
      self.relu =  nn.ReLU()
      # dense layer 1
      self.fc1 = nn.Linear(768,512)
      # dense layer 2 (Output layer)
      self.fc2 = nn.Linear(512,2)
      # softmax activation function
      self.softmax = nn.LogSoftmax(dim=1)
      # define the forward pass
    def forward(self, sent_id, mask):
        # pass the inputs to the model
      cls_hs = self.bert(sent_id, attention_mask=mask)['pooler_output']

      x = self.fc1(cls_hs)
      x = self.relu(x)
      x = self.dropout(x)
      # output layer
      x = self.fc2(x)
      # apply softmax activation
      x = self.softmax(x)
      return x

model = BERT_Arch(bert)
# Defining the hyperparameters (optimizer, weights of the classes and the epochs)

path = 'bert_model_tweets_v3.pt'
#Load in best BERT model
model.load_state_dict(torch.load(path))
print("evaluating")
with torch.no_grad():
  preds = model(test_seq, test_mask)
  print(preds)
  preds = preds.detach().cpu().numpy()

preds = np.argmax(preds, axis = 1)
print(classification_report(test_y, preds))
