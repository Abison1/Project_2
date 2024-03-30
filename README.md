# Disaster Response Pipeline Project

### Overview:

We are provided with two datasets where I built an Extract Transforn Load(ETL), Natural Language Processing(NLP) and Machine Learning(ML) pipelines to categorise messages based on needs required by victim

## Steps in Project: 
1. Create an ETL pipeline. 

    -  Load data and merge two datasets, given files are message.csv and categories.csv
    -  Wrangle the merged dataset to create clean dataset
    -  save dataset as an SQLlite database, this will be input to ML
    -  output is script process.py

2. Create an ML pipeline. 
    -  Load the data from SQL database(from ETL pipeline)
    -  Use a custom tokenize function using nltk to case normalize, lemmatize, and tokenize text.
    -  Use gridsearch to get best parameters
    -  Build and Save mode as a pickle file
    -  output is a script train_classifier.py

3. Create a Flask App
    -  The app will show category of user message
    -  The app will show Visuals from database created from dataset


## Files & folders in project
app
    - templates(contains run.py,go.html and master.html)
        - run.py : A Flask app that will highlight message category as given by user
        - go.html and master.html : web files used for visuals in app
data
    - disaster_categories.csv : csv file that cotains different categories
    - disaster_messages.csv : csv file that contains difderent messages
    - DisasterRespose.dv : an SQL database created given datasets
    - process.py :  a script for ETL

Models
    - train_classier.py : a script for ML
    - model.pkl : model file created from ML script
README.md 
    - this file

## Libraries Used
    * sys
    * pandas
    * nltk
    * Scikit-learn
    * sqlalchemy
    * re
    * pickle
    * json
    * plotly
    * flask

## Results
    To get the app working, you need to run run.py file.
    Below graph shows default page.
    Subsequent graphs shows the top ten and botton message type.

![Immage 1](https://github.com/Abison1/Project_2/blob/main/images/image-1.png?raw=true)
![Image_2](https://github.com/Abison1/Project_2/blob/main/images/image-2.png?raw=true)
