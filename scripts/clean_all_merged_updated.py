import pandas as pd
import re

def clean_text(text):
	if isinstance(text, str):
		# Replace periods with spaces
		text = re.sub(r'\.', ' ', text)
		# Remove links
		text = re.sub(r'http\S+', '', text)
		# Remove mentions (@usernames)
		text = re.sub(r'@[\w]+', '', text)
		# Remove hashtags (#hashtag)
		text = re.sub(r'#\S+', '', text)
		# Remove anything that isn't alphanumeric or spaces
		text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
		# Remove the sentence "featured image ..." at the end
		text = re.sub(r'featured image.*$', '', text, flags=re.IGNORECASE).strip()
		# Set to lowercase
		text = text.lower()
	return text


def map_status(status):
	status_map = {"false": 0, "mostly true": 1, "mostly-true":1, "barely-true":1, "partially true": 1, "half-true":1, "true": 1, "real":1, "1":1, "fake": 0, "mostly-false":0, "pants-fire":0, "0": 0}
	return status_map.get(status.lower(), status)  # Default to 0 


file_path = "all_merged_updated.csv"
df = pd.read_csv(file_path)

df = df.dropna(subset=['text'])


df['text'] = df['text'].apply(clean_text)


df['status'] = df['status'].apply(map_status)

# Drop where text is too long
df = df[df['text'].apply(len) <= 3000]

df.to_csv("all_merged_more_updated.csv", index=False)
