CrediTweet is a dashboard in which users can check if the tweet of their
choosing contains credible or misleading content. CrediTweet utilises a
fine-tuned BERT model to generate the classifications, as well as other models
to provide credibilities for topics, accounts and so on. View the requirements
and steps below to begin using CrediTweet.

Python Version 3.8 was used for the Project, here are the following required
packages to run the program:
tensorflow
Bertopic
Torch
streamlit
scikit-learn
matplotlib

Create a Python ver 3.8 environment using conda and run the code there.

More packages are listed in the Requirements incase any error occurs

1. Training the Classification Model --> ml-bert-v2.py
It is recommended to use the dataset within the dataset folder of "main_dataset_v2.csv"
for training the model.

2.Training the Topic Model --> ml-topic-bert.py
It is recommended to use the dataset within the dataset folder of "main_dataset_v2.csv"
for training the model.

2. Setting up MongoDB
Create a local MongoDB database and add 3 csv files to 3 different tables,
these being data:
accounts--> accounts_v2.csv
topics--> topic_labels_v2.csv
data--> topics_v2_formatted.csv

3. Running the Dashboard --> creditweet.py
Run the creditweet.py file with the command line "streamlit run creditweet.py"
