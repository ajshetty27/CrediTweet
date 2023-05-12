import numpy as np
import pandas as pd
from bertopic import BERTopic
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import os
import pickle5 as pickle
import transformers
from transformers import AutoModel, BertTokenizerFast
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from transformers import AdamW


def predictor(X_test):
    #Load in tweet
    X_test = X_test
    #Load in
    topic_model = BERTopic.load("topic_model")
    pac_model = pickle.load(open('pac_model_v2.pkl', 'rb'))
    tfidf_vectorizer = pickle.load(open('tfidf_vectorizer_v2.pkl', 'rb'))

    def tokenize(X):
          X = tokenizer(
            text = list(X),
            add_special_tokens = True,
            max_length = 100,
            truncation = True,
            padding = 'max_length',
            return_tensors = 'tf',
            return_token_type_ids = False,
            return_attention_mask = True,
            verbose = True
            )
          return X
    bert = AutoModel.from_pretrained('bert-base-uncased')
    tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')

    for param in bert.parameters():
        param.requires_grad = False    # false here means gradient need not be computed

    class BERT_Arch(nn.Module):
        def __init__(self, bert):
          super(BERT_Arch, self).__init__()
          self.bert = bert
          self.dropout = nn.Dropout(0.1)            # dropout layer
          self.relu =  nn.ReLU()                    # relu activation function
          self.fc1 = nn.Linear(768,512)             # dense layer 1
          self.fc2 = nn.Linear(512,2)               # dense layer 2 (Output layer)
          self.softmax = nn.LogSoftmax(dim=1)       # softmax activation function
        def forward(self, sent_id, mask):           # define the forward pass
          cls_hs = self.bert(sent_id, attention_mask=mask)['pooler_output']
                                                    # pass the inputs to the model
          x = self.fc1(cls_hs)
          x = self.relu(x)
          x = self.dropout(x)
          x = self.fc2(x)                           # output layer
          x = self.softmax(x)                       # apply softmax activation
          return x

    model = BERT_Arch(bert)
    optimizer = AdamW(model.parameters(),lr = 1e-5)
    cross_entropy  = nn.NLLLoss()
    epochs = 10

    path = 'bert_model_v3.pt'
    model.load_state_dict(torch.load(path))

    unseen_news_text = [X_test]

    MAX_LENGHT = 15
    tokens_unseen = tokenizer.batch_encode_plus(
        unseen_news_text,
        max_length = MAX_LENGHT,
        pad_to_max_length=True,
        truncation=True
    )

    unseen_seq = torch.tensor(tokens_unseen['input_ids'])
    unseen_mask = torch.tensor(tokens_unseen['attention_mask'])

    with torch.no_grad():
      preds = model(unseen_seq, unseen_mask)
      preds = preds.detach().cpu().numpy()

    preds = np.argmax(preds, axis = 1)
    print(preds)

    predicted_label = " "
    if preds == 0:
        predicted_label = "REAL"
    else:
        predicted_label = "FAKE"
    print(predicted_label)
    #Get TF-IDF values from a saved model and produce the values for the inputted words 
    tfidf_test=tfidf_vectorizer.transform([X_test])
    new_text_vectorized = tfidf_vectorizer.transform([X_test])
    confidence_score = pac_model.decision_function(tfidf_test)
    coef = pac_model.coef_[0]
    positive_coef = coef[0]
    negative_coef = -coef[0]
    feature_names = tfidf_vectorizer.get_feature_names_out()

    if predicted_label == "REAL":
       top_words_idx = positive_coef.argsort()[-10:][::-1]
    else:
       top_words_idx = negative_coef.argsort()[-10:][::-1]
    top_words = [feature_names[idx] for idx in top_words_idx]

    new_text_coef = new_text_vectorized.toarray()[0] * positive_coef if predicted_label == "REAL" else new_text_vectorized.toarray()[0] * negative_coef
    new_text_top_words_idx = new_text_coef.argsort()[-3:][::-1]
    new_text_top_words = [feature_names[idx] for idx in new_text_top_words_idx]

# Print the predicted label and important words
    print("Predicted label:", predicted_label)
    print("Important words for the predicted label:", top_words)
    print("Important words in the new text:", new_text_top_words)

    label = predicted_label
    print(label)

    print("The provided text is "+label)

    num_of_topics = 1
    similar_topics, similarity = topic_model.find_topics(X_test, top_n=num_of_topics);

    print(f'The top {num_of_topics} similar topics are {similar_topics}, and the similarities are {np.round(similarity,2)}')
    similarity_rounded = np.round(similarity,2)

    return label, confidence_score, num_of_topics, similar_topics, similarity_rounded, new_text_top_words
