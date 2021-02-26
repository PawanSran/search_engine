import pandas as pd
import numpy as np
import nltk
nltk.download('stopwords')
nltk.download('punkt')
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem import PorterStemmer
ps=PorterStemmer()

from nltk.corpus import stopwords
import string
from nltk.corpus import stopwords
stp_wrd=set(stopwords.words('english'))
from bs4 import BeautifulSoup
import re
import joblib
import _pickle as cPickle

### Preprocessing and Cleaning-----

## Tokenize
## Converting all to lower case
## Remove HTML tags
## Remove text between [] & ()
## Remove 's
## Remove ""
## Contraction mapping
## Remove punctuations and numbers
## Remove stop words

def preprocess(transcript):
  contraction_mapping = {"ain't": "is not", "aren't": "are not","can't": "cannot", "'cause": "because", "could've": "could have", "couldn't": "could not","didn't": "did not", "doesn't": "does not", "don't": "do not", "hadn't": "had not", "hasn't": "has not", "haven't": "have not","he'd": "he would","he'll": "he will", "he's": "he is", "how'd": "how did", "how'd'y": "how do you", "how'll": "how will", "how's": "how is","I'd": "I would", "I'd've": "I would have", "I'll": "I will", "I'll've": "I will have","I'm": "I am", "I've": "I have", "i'd": "i would","i'd've": "i would have", "i'll": "i will",  "i'll've": "i will have","i'm": "i am", "i've": "i have", "isn't": "is not", "it'd": "it would","it'd've": "it would have", "it'll": "it will", "it'll've": "it will have","it's": "it is", "let's": "let us", "ma'am": "madam","mayn't": "may not", "might've": "might have","mightn't": "might not","mightn't've": "might not have", "must've": "must have","mustn't": "must not", "mustn't've": "must not have", "needn't": "need not", "needn't've": "need not have","o'clock": "of the clock","oughtn't": "ought not", "oughtn't've": "ought not have", "shan't": "shall not", "sha'n't": "shall not", "shan't've": "shall not have","she'd": "she would", "she'd've": "she would have", "she'll": "she will", "she'll've": "she will have", "she's": "she is","should've": "should have", "shouldn't": "should not", "shouldn't've": "should not have", "so've": "so have","so's": "so as","this's": "this is","that'd": "that would", "that'd've": "that would have", "that's": "that is", "there'd": "there would","there'd've": "there would have", "there's": "there is", "here's": "here is","they'd": "they would", "they'd've": "they would have","they'll": "they will", "they'll've": "they will have", "they're": "they are", "they've": "they have", "to've": "to have","wasn't": "was not", "we'd": "we would", "we'd've": "we would have", "we'll": "we will", "we'll've": "we will have", "we're": "we are","we've": "we have", "weren't": "were not", "what'll": "what will", "what'll've": "what will have", "what're": "what are","what's": "what is", "what've": "what have", "when's": "when is", "when've": "when have", "where'd": "where did", "where's": "where is","where've": "where have", "who'll": "who will", "who'll've": "who will have", "who's": "who is", "who've": "who have","why's": "why is", "why've": "why have", "will've": "will have", "won't": "will not", "won't've": "will not have","would've": "would have", "wouldn't": "would not", "wouldn't've": "would not have", "y'all": "you all","y'all'd": "you all would","y'all'd've": "you all would have","y'all're": "you all are","y'all've": "you all have","you'd": "you would", "you'd've": "you would have", "you'll": "you will", "you'll've": "you will have","you're": "you are", "you've": "you have"}
  ## Lower case
  trnscrpt =transcript.lower()
  ## parsing HTML as text
  trnscrpt =BeautifulSoup(trnscrpt,"lxml").text
  ## Removing text between () and []
  trnscrpt =re.sub("[\(\[].*?[\)\]]", "", trnscrpt)
  ## Removing double quotes
  trnscrpt = re.sub('"','',trnscrpt)
  ## Contraction mapping
  trnscrpt = ' '.join([contraction_mapping[t] if t in contraction_mapping 
                       else t for t in trnscrpt.split(" ")])
  ## Removing apostraphe
  trnscrpt = re.sub(r"'s\b","",trnscrpt)
  ## Removing puntuations and special chars and numbers
  trnscrpt = re.sub("[^a-zA-Z]", " ", trnscrpt) 
  ## TOkenizing
  tokenize_trnscrpt = [w for w in trnscrpt.split() if not w in stp_wrd]

  return(tokenize_trnscrpt)

def clean_stemming(sent):
  l2=list()
  l1=list()
  for i in sent:
    l1=list()
    l1=([ps.stem(word) for word in i])
    
    l2.append(l1)

  return(l2)


def to_string(transcripts):
  trnscpt_to_str=list()
  str=""
  for i in list(range(0,len(transcripts))):
    str=""
    str=(" ".join(transcripts[i]))
    trnscpt_to_str.append(str)

  return(trnscpt_to_str)

from sklearn.feature_extraction.text import TfidfVectorizer

def tf_idf(transcripts):
    
	tfidf_vectorizer = TfidfVectorizer()
	tfidf_transcript_matrix = tfidf_vectorizer.fit_transform(to_string(transcripts))
	return(tfidf_transcript_matrix,tfidf_vectorizer.vocabulary_)

    
transcripts_data = pd.read_csv('./dataset/transcripts.csv',header='infer')
inp=transcripts_data['transcript']
oup=transcripts_data['url']

# Cleaning Transcripts
clean_text=list()
clean_text=[preprocess(i) for i in inp]

# Stemmed Output
stemming=clean_stemming(clean_text)

# tfidf Vectorizer
tfidf_transcript_matrix,vocab=tf_idf(stemming)
data_struct = {'vectorizer': tfidf_transcript_matrix}

joblib.dump(stemming, './preprocessed_op.pkl') 
joblib.dump(oup,'./oup.pkl')


with open('storage.bin', 'wb') as f: 
    cPickle.dump(data_struct, f)

with open('storage.bin1', 'wb') as f: 
    cPickle.dump(vocab, f)



