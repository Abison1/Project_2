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
app <br>
    - templates(contains run.py,go.html and master.html) <br>
    - run.py : A Flask app that will highlight message category as given by user <br>
    - go.html and master.html : web files used for visuals in app <br>
    
data <br>
    - disaster_categories.csv : csv file that cotains different categories <br>
    - disaster_messages.csv : csv file that contains difderent messages <br>
    - DisasterRespose.dv : an SQL database created given datasets <br>
    - process.py :  a script for ETL <br>
    
images <br>
    - two images taken when loading the app <br>

Models <br>
    - train_classier.py : a script for ML <br>
    - model.pkl : model file created from ML script <br>

notebooks <br>
    - ETL Pipeline Prep.ipynb: a jupyter notebook for ETL, output will be process file <br>
    - ML Pipeline Preparation.ipynb: a jupyter notebook for Ml, it will output the model file <br>
    
README.md  <br>
    - this file<br>
    
## How to run python scripts
    # To create a processed sqlite db
    python process_data.py disaster_messages.csv disaster_categories.csv DisasterResponse.db
    # where DisasterResponse.db is the user chosen database file name
    
    # To train and save a pkl model
    python train_classifier.py ../data/DisasterResponse.db classifier.pkl
    #where classifier.pkl is user chosen pickle file name 
    
    # To deploy the application locally
    python run.py

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
