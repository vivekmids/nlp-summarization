#!/usr/bin/env python
# coding: utf-8

# In[18]:


#####
# Packages
##### 

import numpy as np
import pandas as pd 
import re #for regex search purposes          
import pickle
cleanfilename = 'cleandata.pkl'
wikihowfile = 'wikihowSep.csv'


# In[19]:


sepdata = pd.read_csv(wikihowfile)


# In[20]:


print ('Read file. Shape is ', sepdata.shape)


# In[21]:


sepdata_v1 = sepdata.dropna(subset=['headline','text'], axis=0).reset_index(drop=True) 


# In[22]:


print ('Removed NAs. Shape is ', sepdata_v1.shape)


# In[23]:


# List of contractions that we will map to 

contraction_mapping = {"ain't": "is not", "aren't": "are not","can't": "cannot", "'cause": "because", "could've": "could have", "couldn't": "could not",

                           "didn't": "did not", "doesn't": "does not", "don't": "do not", "hadn't": "had not", "hasn't": "has not", "haven't": "have not",

                           "he'd": "he would","he'll": "he will", "he's": "he is", "how'd": "how did", "how'd'y": "how do you", "how'll": "how will", "how's": "how is",

                           "I'd": "I would", "I'd've": "I would have", "I'll": "I will", "I'll've": "I will have","I'm": "I am", "I've": "I have", "i'd": "i would",

                           "i'd've": "i would have", "i'll": "i will",  "i'll've": "i will have","i'm": "i am", "i've": "i have", "isn't": "is not", "it'd": "it would",

                           "it'd've": "it would have", "it'll": "it will", "it'll've": "it will have","it's": "it is", "let's": "let us", "ma'am": "madam",

                           "mayn't": "may not", "might've": "might have","mightn't": "might not","mightn't've": "might not have", "must've": "must have",

                           "mustn't": "must not", "mustn't've": "must not have", "needn't": "need not", "needn't've": "need not have","o'clock": "of the clock",

                           "oughtn't": "ought not", "oughtn't've": "ought not have", "shan't": "shall not", "sha'n't": "shall not", "shan't've": "shall not have",

                           "she'd": "she would", "she'd've": "she would have", "she'll": "she will", "she'll've": "she will have", "she's": "she is",

                           "should've": "should have", "shouldn't": "should not", "shouldn't've": "should not have", "so've": "so have","so's": "so as",

                           "this's": "this is","that'd": "that would", "that'd've": "that would have", "that's": "that is", "there'd": "there would",

                           "there'd've": "there would have", "there's": "there is", "here's": "here is","they'd": "they would", "they'd've": "they would have",

                           "they'll": "they will", "they'll've": "they will have", "they're": "they are", "they've": "they have", "to've": "to have",

                           "wasn't": "was not", "we'd": "we would", "we'd've": "we would have", "we'll": "we will", "we'll've": "we will have", "we're": "we are",

                           "we've": "we have", "weren't": "were not", "what'll": "what will", "what'll've": "what will have", "what're": "what are",

                           "what's": "what is", "what've": "what have", "when's": "when is", "when've": "when have", "where'd": "where did", "where's": "where is",

                           "where've": "where have", "who'll": "who will", "who'll've": "who will have", "who's": "who is", "who've": "who have",

                           "why's": "why is", "why've": "why have", "will've": "will have", "won't": "will not", "won't've": "will not have",

                           "would've": "would have", "wouldn't": "would not", "wouldn't've": "would not have", "y'all": "you all",

                           "y'all'd": "you all would","y'all'd've": "you all would have","y'all're": "you all are","y'all've": "you all have",

                           "you'd": "you would", "you'd've": "you would have", "you'll": "you will", "you'll've": "you will have",

                           "you're": "you are", "you've": "you have"}


# In[24]:


def text_cleaner(text):
    # Step 0: Convert to string in case a float or int is found.
    newString = str(text)
    # Step 1: Lower case the text 
    newString = newString.lower()
    # Step 2: Get rid of commas
    #newString = re.sub(r'\([^)]*\)', '', newString)
    # Step 3: Get rid of quotations 
    newString = re.sub('"','', newString)
    # Step 4: get rid of contractions with our contraction mapping 
    newString = ' '.join([contraction_mapping[t] if t in contraction_mapping else t for t in newString.split(" ")])  
    # Step 5: get rid of the \n stuff 
    newString = re.sub(r"'s\n","",newString)
    # Step 6: get rid of most special characters
    newString = re.sub(r'[?$()_\-—\d{}/#&%<>=@\*~:;\\\+\']',r' ', newString) 
    # Step 7: separating punctuations so they are their own tokens
    newString = re.sub(r'([,.!;:?])', r' \1 ', newString)

    return newString.split()


# In[25]:


print("Starting to clean data")


# In[26]:


clean_data = pd.DataFrame()

clean_data['text'] = sepdata_v1['text'].apply(text_cleaner)


# In[27]:


print("Text clean complete. Number of items is ", len(clean_data.text))


# In[28]:


clean_data['headline'] = sepdata_v1['headline'].apply(text_cleaner)


# In[29]:


print("Title cleaning complete. Number of items is ", len(clean_data.headline))


# In[30]:


print('Saving file to disk. File name is ', cleanfilename)


# In[31]:


pickle.dump( clean_data, open( cleanfilename, "wb" ) )


# In[33]:


print('File saved.')


