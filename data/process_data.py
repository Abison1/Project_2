# import required libraries
import sys
import pandas as pd
# import numpy as np
#import sqlite3
#import sqlalchemy
from sqlalchemy import create_engine

def load_data(messages_file, categories_file):
    '''
    This is a function to load data given two files. It create dataframes,
    merges them.Take categories_filepath, name column and change values to 
    0 or 1

    :param messages_filepath: a file with messages
    :param categories_filepath: a file with categories
    :return: df, a combuned dataframe of messages and categories(with 0 or 1 values)
    '''

    # Load data
    messages = pd.read_csv(messages_file)
    categories = pd.read_csv(categories_file)

    # Merge datasets
    df = messages.merge(categories,on='id')
    
    # Create sepearate individual category columns
    categories = df['categories'].str.split(pat =';',expand=True)
    
    # Select the first row of the categories dataframe
    row = categories.iloc[0]
    
    #lamda funtion to remove the last two characters in column row
    category_colnames = row.apply(lambda x: x[:-2])
    
    # rename the columns of `categories`
    categories.columns = category_colnames
    #Iterate through the category columns in dataframef(categories)
    # to keep only the last character of each string (the 1 or 0)
    # and make output an int
    for column in categories:
         categories[column] = categories[column].astype('str').str[-1].astype('int')
     
        
    # drop the original categories column from `df`
    df = df.drop('categories',axis=1)
    
    # concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df,categories], axis=1)
    
    return df
    
def clean_data(df):
    '''
    This is a function to clean dataframe by removing duplicates and 
    and dropping a column that has more that 40% of NaN

    :param df: dataframe to be cleaned
    :return: a clean dataframe
    '''
    # drop duplicates
    df.drop_duplicates(inplace=True)
    # drop column with 'a lot' of missing data. 'a lot is set at 40%.

    threshold = 0.4 *len(df) # set threshold to 40% of the length of column
    df.dropna(axis=1,thresh=threshold, inplace=True)
    
    # filter/excluse rows on 'related ' that are equal to 2 
    df = df[df["related"]!=2]
    return df

def save_data(df,database_filepath):
    '''
    This function takes the dataframe and create an SQL database with named table

    :param df:dataframe to be saved as SQL database
    :return: named SQl database
    '''
    engine = create_engine(f'sqlite:///{database_filepath}')
    df.to_sql('Messages', engine, index=False, if_exists='replace')  


def main():
    if len(sys.argv) == 4:

        messages_file, categories_file, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_file, categories_file))
        df = load_data(messages_file, categories_file)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()