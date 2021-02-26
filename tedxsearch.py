import pandas as pd
import numpy as np
from nltk.stem import PorterStemmer
ps=PorterStemmer()

import pickle
import _pickle as cPickle

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.metrics.pairwise import cosine_similarity

def cosine_similar(search_query, tfidf_transcript_matrix):
	cosine_distance = cosine_similarity(search_query, tfidf_transcript_matrix)
	sim_list = cosine_distance[0]
  
	return sim_list

def return_URL(sim_list,number=1):
  sim_index=list()
  for i in range(number,0,-1):
    max_index=np.argmax(sim_list)
    #sim_list[max_index]=-999
    sim_index.append(max_index)
  
  return(sim_index)



def main(search_query):

    
  with open('oup.pkl', 'rb') as pickle_load:
    oup= pickle.load(pickle_load)
  
  ## ------------------------------------------------------ Importing clean stemmed data---------------------------------------------------------
  
  with open('storage.bin', 'rb') as f:
      data_struct = cPickle.load(f)
    
  transcript_mtrx=data_struct['vectorizer']
  
  with open('storage.bin1', 'rb') as f:
    vocab= cPickle.load(f)
  
    # Loading original vocab
  tfidf_vectorizer = TfidfVectorizer(vocabulary=vocab)
  
    # search query -> tf/idf representation
  search=tfidf_vectorizer.fit_transform([ps.stem(search_query)])

  #Similarity List

  sim_list=cosine_similar(search,transcript_mtrx)
  
  return(oup[int(' '.join([str(i) for i in return_URL(sim_list)]))])

#print(main("women empowerment"))
