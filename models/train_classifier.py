import sys
import numpy as np
import pandas as pd
import pickle

from sqlalchemy import create_engine

from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import classification_report


def load_data(database_filepath):
    """ load the cleaned dataset

    Args:
    database_filepath: the filepath for cleaned dataset

    Returns:
    X: the features
    Y: the labels
    category_names: a list of the categories of the labels

    """
    database_filepath = 'sqlite:///' + database_filepath
    engine = create_engine(database_filepath)
    df = pd.read_sql_table('DisasterResponse', engine)
    X = df['message']
    Y = df.drop(['id', 'message', 'original', 'genre'], axis = 1)
    category_names = Y.columns.tolist()
    return X, Y, category_names


def tokenize(text):
    """ tokenize the message string and clean it

    Args:
    text: the message string

    Returns:
    clean_tokens: the list of clean tokens

    """
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens


def build_model():
    """ build the pipeline from vectorizer, TfidfTransformer to classifier
    Use GridSearchCV to finetune parameters

    Returns:
    cv: the best pipeline of vectorizer, TfidfTransformer to classifier

    """
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize, ngram_range = (1, 2))),
        ('tfidf', TfidfTransformer()),
        ('clf',
     MultiOutputClassifier(RandomForestClassifier(n_estimators = 20)))
    ])

    parameters = {
    'vect__ngram_range': ((1, 1), (1, 2)),
    'clf__estimator__min_samples_split': [2, 3]
    }

    cv = GridSearchCV(pipeline, param_grid=parameters, cv = 3, scoring='f1_marco')

    cv.fit(X_train, Y_train)
    return cv


def evaluate_model(model, X_test, Y_test, category_names):
    """ evaluate the model on test data set

    Args:
    model: the trained model (pipeline)
    X_test: the features of test data set
    Y_test: the labels of test data set
    category_names: a list of the categories of the labels

    Returns:
    # display results


    """
    Y_pred = model.predict(X_test)
    Y_pred = pd.DataFrame(Y_pred, columns = category_names)
    for col in category_names:
        try:
            print(col)
            print(classification_report(Y_test[col], Y_pred[col]))
        except:
            continue


def save_model(model, model_filepath):
    """ save the trained model as a pickle file

    Args:
    model: the trained model (pipeline)
    model_filepath: the filepath where the model is saved
    """
    pickle.dump(model, open(model_filepath, 'wb'))

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
