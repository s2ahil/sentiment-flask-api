from flask import *
import pickle
import sys
import json
from nltk.stem import WordNetLemmatizer
import re      
import pandas as pd

app=Flask(__name__)

@app.route('/',methods=['POST'])
def predict():
  shortcode= request.form.to_dict()
  # Get a list of all the values
  values = list(shortcode.values())
  print(values)
  text=values

  with open('vectoriser-ngram-(1,2).pickle', 'rb') as f:
    vectorizer = pickle.load(f)

  with open('Sentiment-LR.pickle', 'rb') as f:
      model = pickle.load(f)
  
  # Defining dictionary containing all emojis with their meanings.
  emojis = {':)': 'smile', ':-)': 'smile', ';d': 'wink', ':-E': 'vampire', ':(': 'sad', 
          ':-(': 'sad', ':-<': 'sad', ':P': 'raspberry', ':O': 'surprised',
          ':-@': 'shocked', ':@': 'shocked',':-$': 'confused', ':\\': 'annoyed', 
          ':#': 'mute', ':X': 'mute', ':^)': 'smile', ':-&': 'confused', '$_$': 'greedy',
          '@@': 'eyeroll', ':-!': 'confused', ':-D': 'smile', ':-0': 'yell', 'O.o': 'confused',
          '<(-_-)>': 'robot', 'd[-_-]b': 'dj', ":'-)": 'sadsmile', ';)': 'wink', 
          ';-)': 'wink', 'O:-)': 'angel','O*-)': 'angel','(:-D': 'gossip', '=^.^=': 'cat'}


  
  def preprocess(textdata):
    
          processedText = []
          
          # Create Lemmatizer and Stemmer.
          wordLemm = WordNetLemmatizer()
          
          # Defining regex patterns.
          urlPattern        = r"((http://)[^ ]*|(https://)[^ ]*|( www\.)[^ ]*)"
          userPattern       = '@[^\s]+'
          alphaPattern      = "[^a-zA-Z0-9]"
          sequencePattern   = r"(.)\1\1+"
          seqReplacePattern = r"\1\1"
          
          for tweet in textdata:
              tweet = tweet.lower()
              
              # Replace all URls with 'URL'
              tweet = re.sub(urlPattern,' URL',tweet)
              # Replace all emojis.
              for emoji in emojis.keys():
                  tweet = tweet.replace(emoji, "EMOJI" + emojis[emoji])        
              # Replace @USERNAME to 'USER'.
              tweet = re.sub(userPattern,' USER', tweet)        
              # Replace all non alphabets.
              tweet = re.sub(alphaPattern, " ", tweet)
              # Replace 3 or more consecutive letters by 2 letter.
              tweet = re.sub(sequencePattern, seqReplacePattern, tweet)
      
              tweetwords = ''
              for word in tweet.split():
                  # Checking if the word is a stopword.
                  #if word not in stopwordlist:
                  if len(word)>1:
                      # Lemmatizing the word.
                      word = wordLemm.lemmatize(word)
                      tweetwords += (word+' ')
                  
              processedText.append(tweetwords)
              
          return processedText

      
  def predict1(vectoriser, model, text):
      # Predict the sentiment
      textdata = vectoriser.transform(preprocess(text))
      sentiment = model.predict(textdata)
      
      # Make a list of text with sentiment.
      data = []
      for text, pred in zip(text, sentiment):
          data.append((text,pred))
          
      # Convert the list into a Pandas DataFrame.
      df = pd.DataFrame(data, columns = ['text','sentiment'])
      df = df.replace([0,1], ["Negative","Positive"])
      return df
  
 
  
  df = predict1(vectorizer, model, text)
  
  print(df['sentiment'])   
  return df['sentiment'].to_json()

if __name__ =="__main__":
    app.run(debug=True)
