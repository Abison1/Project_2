# import libraries
import sys

import re
#import numpy as np
import pandas as pd

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer

#import sqlalchemy
from sqlalchemy import create_engine

from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

import pickle

def load_data(database_filepath):
    '''
    This function load data from databasefile
    
    :param database_filepath: location and name of database to be used
    :return : Dependant(y) and independant(X) variables
    
    '''
    
    engine = create_engine('sqlite:///' + database_filepath)
    conn = engine.connect()
    df = pd.read_sql_table('Messages', conn)
    X = df['message']
    y = df.drop(columns =['id','message','genre'])
    col_names = df[df.columns[3:]].columns
    
    return X, y, col_names

def tokenize(text):
    '''
    The function breaks a text into smaller components, typically words, phrases, symbols,
    or other meaningful elements, which are called tokens
    
    :param text: text to be tokenized
    :return :tokenized text(lemmatized)
    '''
    # normilize case and remove punctuation
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())
    
    # tokenize text
    words = word_tokenize(text)
    
    #remove stop words
     
    words = [w for w in words if w not in stopwords.words("english")]
    
     #reduce words to their stems
    stemmed = [PorterStemmer().stem(w) for w in words]
    
    tokens = [WordNetLemmatizer().lemmatize(w) for w in stemmed]

    return tokens


def build_model():
    '''
    This function 
    '''
    pass
    # create an ML pipeline
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier()))
        ])
    #Improve model using GridSearchCV
    paramGrid = {
        'clf__estimator__n_estimators': [50, 100],
        'clf__estimator__bootstrap': [True, False]}
    cv = GridSearchCV(pipeline, param_grid=paramGrid,cv=3,n_jobs=1,scoring="accuracy")
    return cv
    

def evaluate_model(model, X_test, Y_test, category_names):
    y_pred = model.predict(X_test)
    
    i = 0
    for col in Y_test.columns:
        print('Feature {}: {}'.format(i+1, col))
        print(classification_report(Y_test.iloc[:, i], y_pred[:, i]))
        i = i + 1
              
    accuracy = (y_pred == Y_test.values).mean()
    print('The model accuracy is {:.3f}'.format(accuracy))


def save_model(model, model_filepath):
    pickle_file_path = "model.pkl"
    with open(pickle_file_path, "wb") as pickle_file:
        pickle.dump(model, pickle_file)


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()