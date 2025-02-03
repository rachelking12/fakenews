import pandas as pd
import nltk
from nltk.corpus import stopwords

stop_words = set(stopwords.words('english'))

all_merged = pd.read_csv('all_merged.csv', index_col=0)
df1 = pd.read_csv('../datasets/datacleaned.csv',skiprows=1)
df2 = pd.read_csv('../datasets/fake_and_real_news.csv', skiprows=1)
df3 = pd.read_csv('../datasets/politifact_cleaned.csv', skiprows=1)

df1.columns = ['text', 'status']
df2.columns = ['text', 'status']
df3.columns = ['text', 'status']

all_merged = pd.concat([all_merged, df1, df2, df3], ignore_index=True)

def remove_stopwords(text):
	if isinstance(text, str):  
		return ' '.join(word for word in text.split() if word.lower() not in stop_words)
	return text

# Apply the function to a specific column (change 'text_column' to the actual column name)
all_merged['text'] = all_merged['text'].apply(remove_stopwords)

# Save the updated dataset
all_merged.to_csv('all_merged_updated.csv', index=False, quoting=1)