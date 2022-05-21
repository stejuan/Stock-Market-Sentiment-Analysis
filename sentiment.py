from cv2 import dft
import dash
import dash_bootstrap_components as dbc
from dash import Dash, html, dcc, Input, Output, dash_table
import nltk
from nltk.corpus import stopwords
from nltk import word_tokenize
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import re
from gensim.models import Word2Vec
import plotly.express as px
import xgboost as xgb
# from wordcloud import WordCloud

stock_data = pd.read_csv('stock_data.csv')
newdata = stock_data

for i in range(stock_data.shape[0]): # Remove special characters
    newdata.iloc[i,0]=re.sub('[^A-Za-z ]+', '', newdata.iloc[i,0])
newdata["Tokenized"] = [word_tokenize(line) for line in newdata['Text']] 
stop_words = set(stopwords.words('english'))

def remove_stop(s):
    return [w for w in s if not w.lower() in stop_words]

newdata["Tokenized"] = [remove_stop(line) for line in newdata['Tokenized']]

X = newdata["Tokenized"]
y = newdata["Sentiment"]

from sklearn.model_selection import train_test_split 
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=13)

l = []
for line in X_train:
    l.append(line)

model = Word2Vec(sentences=l, vector_size=1000, window=5, min_count=1, workers=4)

def getVectors(dataset):
  vectors=[]
  for dataItem in dataset:
    wordCount=0
    singleDataItemEmbedding=np.zeros(1000)
    for word in dataItem:
        try:
            singleDataItemEmbedding=singleDataItemEmbedding+model.wv[word]
            wordCount=wordCount+1
        except:
            pass
    singleDataItemEmbedding=singleDataItemEmbedding/wordCount  
    vectors.append(singleDataItemEmbedding)
  return vectors

X_train=getVectors(X_train)
X_test=getVectors(X_test)
X_train = np.nan_to_num(X_train)
X_test = np.nan_to_num(X_test)

xgbmodel = xgb.XGBClassifier()
xgbmodel.fit(X_train,y_train)
preds = xgbmodel.predict(X_test)

from sklearn.ensemble import RandomForestClassifier
m = RandomForestClassifier(n_estimators=250,max_depth=None)
m.fit(X_train,y_train)
predictions = m.predict(X_test)

from sklearn.metrics import classification_report, plot_confusion_matrix
print(classification_report(y_test,preds)) # XGB
print(classification_report(y_test,predictions)) #RF

one = plot_confusion_matrix(m, X_test, y_test)
two = plot_confusion_matrix(xgbmodel, X_test, y_test) 
one.figure_.savefig('cmatrix_rf.png',dpi=300)
two.figure_.savefig('cmatrix_xgb.png',dpi=300)

df = pd.read_csv('stock_data.csv')
card1 = dbc.Card([
    html.H2(children="The Data"),
    html.P(children="The dataset used in the project is the Stock-Market Sentiment Dataset from Kaggle: https://www.kaggle.com/datasets/yash612/stockmarket-sentiment-dataset"),
    html.P(children="Below is a brief snapshot of the dataset:"),
    dash_table.DataTable(df.head().to_dict('records'), [{"name": i, "id": i} for i in df.head().columns])],body=True,)

fig1 = px.histogram(df,x='Sentiment',color='Sentiment')

card2 = dbc.Card([
    html.H2(children="Sentiment Distribution"),
    dcc.Graph(
        id='example-graph',
        figure=fig1)
],body=True,)

app = dash.Dash(external_stylesheets=[dbc.themes.BOOTSTRAP])
app.layout = dbc.Container(
    [
     html.H1(children='Stock Market Tweet Sentiment Analysis',style={'textAlign':'center'}),
     html.H3(children='By: Steven Chung',style={'textAlign':'center'}),
     html.P(children="This project utilizes word embeddings (Word2Vec) and xgboost and random forest models to classify and predict tweet sentiment.",style={'textAlign':'center'}),
     
     dbc.Row([card1]),
     dbc.Row([
         html.H2(children="Exploratory Data Analysis"),
         dbc.Row([card2]),
         dbc.Row([
             dbc.Col([
                 html.H2(children="Word Cloud for Negative Sentiment"),
                 html.Img(src=app.get_asset_url('wc negative.png'))
             ]),
             dbc.Col([
                 html.H2(children="Word Cloud for Positive Sentiment"),
                 html.Img(src=app.get_asset_url('wc positive.png'))
             ])
         ])
         ]),
    dbc.Row([
        html.H2(children="Results"),
        html.P(children="First, we preprocess the data by removing any special characters (e.g. hashtags) and stop words (e.g. and, or, in, etc.). Next, we tokenize the text and use it to train a Word2Vec model in order to create word embeddings. For each tweet, we take the 'mean vector' of each word embedding in the tweet. After splitting the data into training and test sets, we can finally build our classification model (Random Forest). a visualization of the confusion is shown below:"),
        html.H3(children="Confusion Matrix for Random Forest Model"),
        html.Img(src=app.get_asset_url('cmatrix_rf.png')),
        html.P(children="We obtain an accuracy of about 68%, which isn't too shabby!"),
        html.H3(children="Confusion Matrix for XGBoost Model"),
        html.Img(src=app.get_asset_url('cmatrix_xgb.png')),
        html.P(children="We obtain an accuracy of about 69%, which is slightly better.")
    ])

    ],fluid=True,)

if __name__ == '__main__':
    app.run_server(debug=True)
