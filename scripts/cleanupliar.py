import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

df_path = "datasets/train.tsv"
df_valid = "datasets/valid.tsv"

df = pd.read_csv(df_path, sep='\t', header=None, names=[
    "id", "label", "claim", "category", "person", "position", "state", "party", 
    "pants_on_fire", "false", "barely_true", "half_true", "mostly_true", "context"
])
df2 = pd.read_csv(df_valid, sep='\t', header=None, names=[
    "id", "label", "claim", "category", "person", "position", "state", "party", 
    "pants_on_fire", "false", "barely_true", "half_true", "mostly_true", "context"
])

# Map labels to the desired values
label_mapping = {
    "pants_on_fire": "False",
    "barely_true": "True",
    "false": "False",
    "true": "True",
    "half-true": "Partially True",
    "mostly-true": "Mostly True"
}
df["label"] = df["label"].map(label_mapping).fillna(df["label"])
df2["label"] = df2["label"].map(label_mapping).fillna(df["label"])

print(df.head())
print(df2.head())

stop_words = set(stopwords.words('english'))

def remove_stop_words(sent):
  # Split the sentence into individual words 
  words = sent.split() 
  
  # Use a list comprehension to remove stop words 
  filtered_words = [word for word in words if word not in stop_words] 
  
  # Join the filtered words back into a sentence 
  return ' '.join(filtered_words)

df["claim"] = df["claim"].apply(remove_stop_words)
df2["claim"] = df2["claim"].apply(remove_stop_words)

# Desired columns
final_df = df[["claim", "label"]]
final_df2 = df2[["claim", "label"]]


# Save new dataset
final_tsv_path = 'datasets/test_trainfinal.csv'
final_valid_path = 'datasets/test_validfinal.csv'

final_df.to_csv(final_tsv_path, index=False, header=["text", "status"])
final_df2.to_csv(final_valid_path, index=False, header=["text", "status"])

