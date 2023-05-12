import pandas as pd
import numpy as np
from bertopic import BERTopic
from bertopic.representation import MaximalMarginalRelevance
import os
from sklearn.feature_extraction.text import CountVectorizer


os.environ["TOKENIZERS_PARALLELISM"] = "false"

df=pd.read_csv('main_dataset.csv')
text_list = df["title"].tolist()

print("Running...")

vectorizer_model = CountVectorizer(stop_words="english")
representation_model = MaximalMarginalRelevance(diversity=0.3)

topic_model = BERTopic(representation_model=representation_model,vectorizer_model=vectorizer_model, language="multilingual", n_gram_range=(1, 3),nr_topics=100)

topics, probs = topic_model.fit_transform(text_list)

topic_labels = topic_model.generate_topic_labels(nr_words= 2,
                                                 topic_prefix=False,
                                                 separator=" , ")

topic_model.set_topic_labels(topic_labels)

df_topic_labels = topic_model.get_topic_info()
df_topics = topic_model.get_document_info(text_list)



df_topic_main = pd.DataFrame(columns=['Text', 'Topic', 'Label'])
df_topic_labels_main = pd.DataFrame(columns=['Topic','Topic Name','Total Count','Total Real','Total Fake'])

print(df_topics)
print(df_topic_labels)
print()
print()
print("Running pt 2......")
topic_model.save("topic_model")

df_topics.to_csv("topics_v2.csv")
df_topic_labels.to_csv("topic_labels_v2.csv")


'''for ind_x in df_topic_labels.index:
    topic = df_topic_labels['Topic'][ind_x]
    topic_name = str(df_topic_labels['CustomName'][ind_x])
    total_count = df_topic_labels['Count'][ind_x]
    real_count = 0
    fake_count = 0

    for ind in df_topics.index:
        text = str(df_topics['Document'][ind])
        topic_2 = df_topics['Topic'][ind]
        label = str(df['tag'][ind])
        if topic == topic_2 and label == "FAKE":
            fake_count +=1
        elif topic == topic_2 and label == "REAL":
            real_count +=1

        df_topic_main = df_topic_main.append({'Text':text,'Topic':topic,'Label':label}, ignore_index = True)

    df_topic_labels_main = df_topic_labels_main.append({'Topic':topic,'Topic Name':topic_name,'Total Count':total_count,'Total Real':real_count,'Total Fake':fake_count}, ignore_index = True)
    print(df_topic_labels_main)

df_topic_main.to_csv('main_dataset_topics.csv')
df_topic_labels_main.to_csv('topic_label_dataset.csv')'''
