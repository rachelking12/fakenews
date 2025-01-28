import pandas as pd
import re
import nltk
from nltk.corpus import stopwords

isot_true = pd.read_csv('datasets/isot/Fake.csv')
isot_fake = pd.read_csv('datasets/isot/True.csv')

stop_words = set(stopwords.words('english'))

def rem_stop_words(sent):
    words = sent.split()
    filtered_words = [word for word in words if word not in stop_words] 
    return ' '.join(filtered_words)

def rem_weird_text(text):
    pattern = r"^[^-]+ - "
    # Using re.sub() to replace the matched pattern with an empty string
    return re.sub(pattern, "", text)

fake_df = pd.DataFrame()
fake_df['text'] = isot_fake["text"].apply(rem_stop_words)
fake_df['text'] = fake_df['text'].apply(rem_weird_text)
fake_df['status'] = 'false'

true_df = pd.DataFrame()
true_df['text'] = isot_true['text'].apply(rem_stop_words)
true_df['status'] = 'true'

fake_df = pd.concat([fake_df, true_df])
fake_df.to_csv('datasets/cleaned/isot_cleaned.csv')