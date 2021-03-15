import sys
import pandas as pd
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    """ load messages dataset and categories dataset; merge the messages and
    categories datasets using the common id

    Args:
    messages_filepath: the filepath for messages dataset
    categories_filepath: the filepath for categories dataset

    Returns:
    The combined dataset of the messages dataset and categories dataset

    """
    messages = pd.read_csv('disaster_messages.csv')
    categories = pd.read_csv('disaster_categories.csv')
    df = messages.merge(categories,on=['id'], how='inner')
    return df

def clean_data(df):
    """ clean combined dataset of the messages dataset and categories dataset

    Args:
    df: the combined dataset

    Returns:
    df: the cleaned combined dataset

    """

    # create a dataframe of the 36 individual category columns
    categories = df['categories'].str.split(';', expand = True)
    # select the first row of the categories dataframe
    row = categories.iloc[1]
    # use this row to extract a list of new column names for categories.
    category_colnames = [s.split('-')[0] for s in categories.iloc[1]]
    # rename the columns of `categories`
    categories.columns = category_colnames

    # create one-hot encodings for categories
    for column in categories:
        categories[column] = categories[column].str.split('-').str.get(1).astype(int)
    categories = categories.applymap(lambda x:1 if x > 0 else x)

    # drop the original categories column from `df`
    del df['categories']

    # concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df, categories], axis=1)

    # drop duplicates
    df.drop_duplicates(inplace=True)

    return df


def save_data(df, database_filename):
    """ save the clean combined dataset to database

    Args:
    df: the cleaned combined dataset
    database_filename: the filepath for database

    """
    database_filename = 'sqlite:///' + database_filename
    engine = create_engine(database_filename)
    df.to_sql('DisasterResponse', engine, index=False)


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

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
