#!/usr/bin/env python
# coding: utf-8

# In[1]:


import requests # to get the website
import re # to grab the exact element we want
import time # to force our code to wait a little before re-trying to grab a webpage
from bs4 import BeautifulSoup # to grab the html that we need
import os # change python working directory

# In[2]:
## check the working directory 
os.getcwd()

# scrape web content 
## create an empty list
movie_data  = [] 
# access the webpage as Chrome
my_headers = { 'User-Agent': 'Mozilla/5.0 (Windows NT 6.1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/41.0.2228.0 Safari/537.36'}

## set up movie and page numbers  
movie = ['frozen_ii', 'joker_2019', 
        'ford_v_ferrari', 'doctor_sleep', 
        'us_2019','black_panther_2018', 
        'dunkirk_2017','baby_driver',
        'gemini_man_2019','the_lion_king_2019', 'maleficent_mistress_of_evil']

pageNum = 20

## scrape 10 pages 
for m in range(0,11):
    for k in range(1,pageNum+1):
        page = 'https://rottentomatoes.com/m/' + movie[m] + '/reviews?page=' + str(k) +'/'
        ## set src as false
        src = False
        ## try 5 times to scrape web content
        for i in range(1,6):
            try:
                ## get url content
                response = requests.get(page, headers=my_headers)
                ## get the html content
                src = response.content
                ## if successfully got the file, break the loop
                break
            except:
                print('failed attempt #', i)
                ## wait 2 secs before trying again
                time.sleep(2)
        ## if we could not get the page
        if not src:
            ## couldn't get the page, print that we could not get page
            print('Could not get page', page)
            continue
        else:
            ## got the page
            print('Successfully got page', page)

        soup = BeautifulSoup(src.decode('ascii','ignore'),'lxml')

        divs = soup.findAll('div', {'class':re.compile('review_table_row')})    
        for div in divs:
            ## initialize to not found
            name = 'NA'
            source = 'NA'
            rating = 'NA'
            content_text = 'NA'
            date_text = 'NA'
            # find a
            a = div.find('a', {'class':re.compile('unstyled bold articleLink')})
            # find em
            em = div.find('em',{'class':re.compile('subtle critic-publication')})
            # find div
            review = div.find('div', {'class':re.compile('review_icon icon small')})
            content = div.find('div', {'class':re.compile('the_review')})
            date = div.find('div', {'class':re.compile('review-date subtle small')})
            ## if find a, grab the text and strip it
            if a:
                name = a.text.strip()
            ## if find em, grab the text and strip it 
            if em:
                source = em.text.strip()
            ## if find review, grab the attribute class
            if review:
                rating = review.attrs['class'][3]
            ## if find content, grab the text and strip it
            if content:
                content_text = content.text.strip()
            ## if find date, grab the text and strip it
            if date:
                date_text = date.text.strip()
            ## add all the info to the data link
            movie_data.append([name,rating,source,content_text,date_text])

with open('movie.txt', mode='w', encoding='utf-8') as f:
    for text in movie_data:
        f.write(text[0] + '\t' + text[1] + '\t' + text[2] + '\t' + text[3]+ '\t' + text[4] + '\n')

# In[4]:
# Text Processing
## Import nltk & string
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
nltk.download('stopwords')
import string
from string import punctuation

## Import pandas to read in data
import pandas as pd
import numpy as np

data = pd.read_csv('movie.txt', delimiter='\t', header=None)
data.columns = ['name', 'rates', 'source', 'content', 'date']
data.head()

## check the null and na
data.isna().any()
data.isnull().any()

## convert text to lowercase
for i in range(len(data)):
    data.iloc[i]['content'] = ' '.join(word.lower() for word in nltk.word_tokenize(data.iloc[i]['content']))

## remove punctuation
for i in range(len(data)):
    data.iloc[i]['content'] = ' '.join(word for word in nltk.word_tokenize(data.iloc[i]['content']) if word not in punctuation)

## stopword remove
stopword = stopwords.words('english')
for i in range(len(data)):
    data.iloc[i]['content'] = ' '.join(word for word in nltk.word_tokenize(data.iloc[i]['content']) if word not in stopword)

## Lemmatizer - aiming to remove inflectional endings only and to return the base or dictionary form of a word
word_lemmatizer = WordNetLemmatizer()
for i in range(len(data)):
     data.iloc[i]['content'] = ' '.join(word_lemmatizer.lemmatize(word) for word in nltk.word_tokenize(data.iloc[i]['content']))

with open('text_process.txt', 'w', encoding="utf-8") as f:
    for i in range(len(data)):
        f.write(data.iloc[i]['name'] + '\t' + data.iloc[i]['content'] + '\t' + data.iloc[i]['rates'] + '\t' + data.iloc[i]['date'] +'\n')

# Modeling
## import plotting
import matplotlib.pylab as plt
import seaborn as sns
newData = pd.read_csv('text_process.txt', delimiter='\t', header=None)
newData.columns = ['name', 'review', 'rates', 'date']
newData.head() 

## set dataset index to time
newData = newData.set_index('date')
newData.index
newData.head()

## Import models and evaluation functions
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_validate
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC

## Import vectorizers to turn text into numeric
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer 

X_text = newData['review']
Y = newData['rates']

## create vectorizer
count_vectorize = CountVectorizer()
## let the vectorizer learn the token
count_vectorize.fit(X_text)
## turn the token into numeric matrix
X = count_vectorize.transform(X_text) 

## create model - random forest
## try to tune max_depth
max_depths = range(10,110,10) # the maximum depth of each tree
accuracies = []
for max_depth in max_depths: 
    param_grid = {
        'max_depth': [max_depth],
        'n_estimators': [10]
    }
    ## create empty unlearned random forest model
    random_forest = RandomForestClassifier()
    ## tune the model and get 10-fold cross validation results 
    grid_search = GridSearchCV(random_forest, param_grid=param_grid, cv=10)
    ## fit and train the random forest
    grid_search.fit(X,Y)
    ## get cv results and keep tracking them
    accuracies.append(grid_search.best_score_)

## plot the results
plt.plot(max_depths, accuracies)
plt.show()

## model - svm
C = [0.1,1,10,100,1000]
accuracies2 = []
for c in C:
    # create grid for model tuning
    param_grid2 = {
        'C': [c]
    } 
    ## create empty unlearned svm model
    svm = SVC()
    ## tune the model and get 10-fold cross validation results 
    grid_search2 = GridSearchCV(svm, param_grid=param_grid2, cv=10)
    ## fit and train the random forest
    grid_search2.fit(X,Y)
    ## get cv results and keep tracking them
    accuracies2.append(grid_search2.best_score_)

## plot the results
plt.plot(C, accuracies2)
plt.show()

performance_svm = max(accuracies2)
performance_random_forest = max(accuracies)

## Improve the model
## create grid for model tuning
gammas = [1,0.1,0.01,0.001,0.0001]
accuracies3 = []
for gamma in gammas:
    param_grid3 = {
        'C': [grid_search2.best_params_['C']],
        'gamma': [gamma]
    } 
    ## create empty unlearned svm model
    svm2 = SVC()
    ## tune the model and get 10-fold cross validation results 
    grid_search3 = GridSearchCV(svm2, param_grid=param_grid3, cv=10)
    ## fit and train the random forest
    grid_search3.fit(X,Y)
    # get the accuracy and keep tacking them
    accuracies3.append(grid_search3.best_score_)

performance_svm2 = max(accuracies3)

## plot the results
plt.plot(gammas, accuracies3)
plt.show()

## compare the result within three models
performanceDf = pd.DataFrame([performance_random_forest, 
                    performance_svm, performance_svm2], 
                    index=['rf','svm','svm2'], columns=['performance']) 
performanceDf