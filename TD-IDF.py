#passage corpus into a DataFrame using pd.read_csv()
#`apply()` used for tokenize/lowercase/stopword removal/stem text 
#for each passage in the DataFrame
#loop over the set of all terms in the passage corpus for IDF
#loop over the passages and the terms in each passage for TD-IDF
#`sorted()` used to rank documents by total TD-IDF score


#Text Preprocessing
#Import libraries
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from collections import Counter
import math
import pandas as pd

# Load the DataFrame containing the passage corpus -- change name of file
passage_df = pd.read_csv('passage_corpus.csv')

#Tokenize the text
passage_df['tokens'] = passage_df['passage'].apply(nltk.word_tokenize)

#Lowercase the tokens
passage_df['tokens'] = passage_df['tokens'].apply(lambda x: [token.lower() for token in x])

#Remove stopwords
stop_words = set(stopwords.words('english'))
passage_df['tokens'] = passage_df['tokens'].apply(lambda x: [token for token in x if token not in stop_words])

#Stem the tokens
stemmer = PorterStemmer()
passage_df['tokens'] = passage_df['tokens'].apply(lambda x: [stemmer.stem(token) for token in x])

#Term Frequency Calculation
passage_df['tf'] = passage_df['tokens'].apply(Counter)

#Inverse Document Frequency Calculation
#Calculate the IDF for each term

num_docs = len(passage_df)
idf = {}
for term in set([term for tf_dict in passage_df['tf'] for term in tf_dict]):
    doc_count = sum(1 for tf_dict in passage_df['tf'] if term in tf_dict)
    idf[term] = math.log(num_docs / doc_count)

#TD-IDF Calculation
#Calculate the TD-IDF score for each term in each document

tdidf = {}
for i, tf_dict in enumerate(passage_df['tf']):
    tdidf[i] = {}
    for term in tf_dict:
        tfidf = tf_dict[term] * idf[term]
        tdidf[i][term] = tfidf

# Rank the documents by TD-IDF score
ranked_docs = sorted(tdidf.items(), key=lambda x: sum(x[1].values()), reverse=True)

