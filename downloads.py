import nltk
import os

nltk.download('stopwords')
nltk.download('wordnet') 
nltk.download('averaged_perceptron_tagger')

os.system('python -m spacy download en_core_web_sm')