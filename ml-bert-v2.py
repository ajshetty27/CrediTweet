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
batch_size = 32
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

# Define the optimizer
optimizer = AdamW(model.parameters(), lr = 1e-5)  # learning rate
# Define the loss function
cross_entropy  = nn.NLLLoss()
# Number of training epochs
epochs = 10

def train():
  model.train()
  total_loss, total_accuracy = 0, 0
  # iterate over batches
  for step,batch in enumerate(train_dataloader):
    if step % 50 == 0 and not step == 0:
    # print progress update after every 50 batches.
      print('  Batch {:>5,}  of  {:>5,}.'.format(step, len(train_dataloader)))
       # push the batch to gpu
    batch = [r for r in batch]
    sent_id, mask, labels = batch
    # clear previously calculated gradients
    model.zero_grad()
    # get model predictions for current batch
    preds = model(sent_id, mask)
    # generate loss between actual & predicted values
    loss = cross_entropy(preds, labels)
    # add on to the total loss
    total_loss = total_loss + loss.item()
    # backward pass to calculate the gradients
    loss.backward()
    # clip gradients to 1.0. It helps in preventing exploding gradient problem
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    # update parameters
    optimizer.step()
    # model predictions are stored on GPU. So, push it to CPU
    preds=preds.detach().cpu().numpy()
  # compute training loss of the epoch
  avg_loss = total_loss / len(train_dataloader)
 # returns the loss and predictions
  return avg_loss

def evaluate():
  print("\nEvaluating...")
  # Deactivate dropout layers
  model.eval()
  total_loss, total_accuracy = 0, 0
  # Iterate over batches
  for step,batch in enumerate(val_dataloader):
     # Progress update every 50 batches.
    if step % 50 == 0 and not step == 0:
      # Calculate elapsed time in minutes.
      # Elapsed = format_time(time.time() - t0)
      # Report progress
      print('  Batch {:>5,}  of  {:>5,}.'.format(step, len(val_dataloader)))
     # Push the batch to GPU
    batch = [t for t in batch]

    sent_id, mask, labels = batch
    # Deactivate autograd
    with torch.no_grad():
      # Model predictions
      preds = model(sent_id, mask)
      # Compute the validation loss between actual and predicted values
      loss = cross_entropy(preds,labels)
      total_loss = total_loss + loss.item()
      preds = preds.detach().cpu().numpy()
      # compute the validation loss of the epoch
  avg_loss = total_loss / len(val_dataloader)
  return avg_loss

best_valid_loss = float('inf')
# empty lists to store training and validation loss of each epoch
train_losses=[]
valid_losses=[]

for epoch in range(epochs):
    print('\n Epoch {:} / {:}'.format(epoch + 1, epochs))
    # train model
    train_loss = train()
    # evaluate model
    valid_loss = evaluate()
    if valid_loss < best_valid_loss:
        # save the best model
        best_valid_loss = valid_loss
        torch.save(model.state_dict(), 'bert_model_v3.pt')
    # append training and validation loss
    train_losses.append(train_loss)
    valid_losses.append(valid_loss)

    print(f'\nTraining Loss: {train_loss:.3f}')
    print(f'Validation Loss: {valid_loss:.3f}')

path = 'bert_model_v3.pt'
#Load in best BERT model
model.load_state_dict(torch.load(path))

with torch.no_grad():
  preds = model(test_seq, test_mask)
  preds = preds.detach().cpu().numpy()

preds = np.argmax(preds, axis = 1)
print(classification_report(test_y, preds))
