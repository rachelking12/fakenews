import pandas as pd
import re

def load_data(filepath):
    df = pd.read_csv('/home/mohit/fakenews/scripts/all_merged.csv')
    return df[:40000]

def convert_status(status):
    if status.lower() == 'true':
        return 1
    else:
        return 0

def clean_text(text):
    if isinstance(text, str):
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        return text
    else:
        return "" # or some other default value



df = load_data('sus')
df = df.dropna(subset=['text'])
df['text'] = df['text'].astype(str)
df['text'] = df['text'].fillna('')
df['text'] = df['text'].astype(str)
df['status'] = df['status'].apply(convert_status)
df['text'] = df['text'].apply(clean_text)
print(df['status'].unique())
df.to_csv('40ktf.csv', index=False)
