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
            data.append({'title': title, 'content': content, 'status': 'fake' if 'fake' in path else 'true'})
    return pd.DataFrame(data)


fake_path = 'datasets/fakeNewsDatasets/celebrityDataset/fake'
legit_path = 'datasets/fakeNewsDatasets/celebrityDataset/legit'

fake_df = loop_dir(fake_path)
legit_df = loop_dir(legit_path)

combined_df = pd.concat([fake_df, legit_df])
combined_df.to_csv('datasets/combined_news.csv', index=False)