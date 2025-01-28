# TEXT ######### STATUS
#

import nltk

import pandas as pd

isot_data = pd.read_csv('datasets/isot_cleaned.csv')

temp_df = pd.DataFrame()

temp_df['text'] = isot_data['text']
temp_df['status'] = isot_data['status']

import os
import pandas as pd
import nltk
from nltk.corpus import stopwords

stop_words = set(stopwords.words('english'))

def remove_stop_words(sent):
  # Split the sentence into individual words 
  words = sent.split() 
  
  # Use a list comprehension to remove stop words 
  filtered_words = [word for word in words if word not in stop_words] 
  
  # Join the filtered words back into a sentence 
  return ' '.join(filtered_words)


def loop_dir(path):
    data = []
    for filename in os.listdir(path):
        filepath = os.path.join(path, filename)
        with open(filepath, 'r', encoding='utf-8') as file:
            lines = file.readlines()
            title = lines[0].strip()
            content = ' '.join([line.strip() for line in lines[1:]]).strip()
            data.append({'text': content, 'status': 'fake' if 'fake' in path else 'true'})
    return pd.DataFrame(data)


fake_path = 'datasets/fakeNewsDatasets/celebrityDataset/fake'
legit_path = 'datasets/fakeNewsDatasets/celebrityDataset/legit'

fake_df = loop_dir(fake_path)
legit_df = loop_dir(legit_path)

combined_df = pd.concat([fake_df, legit_df])

combined_df["text"] = combined_df["text"].apply(remove_stop_words)
temp_df = pd.concat([temp_df, combined_df])

import os
import pandas as pd

def loop_dir(path):
    data = []
    for filename in os.listdir(path):
        filepath = os.path.join(path, filename)
        with open(filepath, 'r', encoding='utf-8') as file:
            lines = file.readlines()
            title = lines[0].strip()
            content = ' '.join([line.strip() for line in lines[1:]]).strip()
            data.append({'text': content, 'status': 'fake' if 'fake' in path else 'true'})
    return pd.DataFrame(data)

print(len(temp_df))

fake_path = 'datasets/fakeNewsDatasets/fakeNewsDataset/fake'
legit_path = 'datasets/fakeNewsDatasets/fakeNewsDataset/legit'

fake_df = loop_dir(fake_path)
legit_df = loop_dir(legit_path)

combined_df["text"] = combined_df["text"].apply(remove_stop_words)
temp_df = pd.concat([temp_df, combined_df])


liar_train = pd.read_csv('datasets/test_trainfinal.csv')
liar_valid = pd.read_csv('datasets/test_validfinal.csv')

temp_df = pd.concat([temp_df, liar_train])
temp_df = pd.concat([temp_df, liar_valid])

temp_df.to_csv('out.csv')