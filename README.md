# disaster-response-pipelines
This project analyzes disaster data from [Figure Eight](https://www.figure-eight.com/) to build a model for an API that classifies disaster messages. The data set contains real messages that were sent during disaster events. We will be creating a machine learning pipeline to categorize these events so that we can send the messages to an appropriate disaster relief agency.

## Table of Contents
1. [ETL Pipeline](#etl)
2. [ML Pipeline](#ml)
3. [Flask Web App](#web)
4. [License](#license)

<a name="etl"></a>
## ETL Pipeline
The first part of the data pipeline is the Extract, Transform, and Load process. Here, we read the dataset, clean the data, and then store it in a SQLite database. To load the data into an SQLite database, we use the pandas dataframe .to_sql() method.

EDA is performed in order to figure out how we want to clean the data set. 

The Python script, process_data.py, includes a data cleaning pipeline that:

- Loads the messages and categories datasets
- Merges the two datasets
- Cleans the data
- Stores it in a SQLite database

<a name="ml"></a>
## ML Pipeline
The Python script, train_classifier.py, includes a machine learning pipeline that:

- Loads data from the SQLite database
- Splits the dataset into training and test sets
- Builds a text processing and machine learning pipeline
- Trains and tunes a model using GridSearchCV
- Outputs results on the test set
- Exports the final model as a pickle file

<a name="web"></a>
### Flask Web App
This project includes a web app where an emergency worker can input a new message and get classification results in several categories. The web app also displays visualizations of the data. 

<a name="license"></a>
## License
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
