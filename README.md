# Stock-Market-Sentiment-Analysis
By: Steven Chung

## Overview

As someone who recently got into the stock market in mid-2021, I decided to intersect my recently-gained knowledge in NLP/ML (i.e. word embeddings, classification, text preprocessing, etc.) and finance in order to conduct sentiment analysis on stock market tweets. 

I was curious to see if I could build a reasonably accurate classifier and find patterns between the semantics/meaning of the tweets and their corresponding sentiment.

## Process

### Dataset

I used the Stock Market Tweet Dataset from Kaggle which can be found [here](https://www.kaggle.com/datasets/yash612/stockmarket-sentiment-dataset?select=stock_data.csv). 

### Exploratory Data Analysis

I conducted some exploratory data analysis by creating some charts to visualize the distribution of sentiment, along with word clouds to provide some visual insight on the types of text prevalent in each sentiment.

### Text Preprocessing

Perhaps the most tedious part of this project, I conduced some text preprocessing in order to train a Word2Vec word embedding model. This involved removing stop words and special characters from the text data.

### Model

After splitting the data into a training and test set, text preprocessing, and creating word embeddings, I trained a few classification models (Random Forest, Decision tree, and XGBoost). They all had similar predictive accuracy, with XGBoost coming in with the highest predictive accuracy at 72%. 

### Dash Web Application

I created a web application using Python's Dash library in order to visualize the results from my exploratory data analysis and model evaluation. 

### Conclusion and Remarks

I hope you enjoy reading this project! I'm still learning and am not perfect so if there are any improvements that can be made to my code or methodology, please let me know! I am always open to learn and improve.

### Tools/Libraries Used
Python, NumPy, sklearn, pandas, NLTK, XGBoost, Gensim, Regex, Dash, Dash Bootstrap Components, wordcloud
