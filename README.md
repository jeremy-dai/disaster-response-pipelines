# disaster-response-pipelines
This project analyzes disaster data from [Figure Eight](https://www.figure-eight.com/) to build a model for an API that classifies disaster messages. The data set contains real messages that were sent during disaster events. We will be creating a machine learning pipeline to categorize these events so that we can send the messages to an appropriate disaster relief agency.

## Table of Contents
1. [Files in the repository](#files)
2. [ETL Pipeline](#etl)
3. [ML Pipeline](#ml)
4. [Flask Web App](#web)
5. [License](#license)

<a name="files"></a>
## Files in the repository
app<br/>
| - template<br/>
| |- master.html # main page of web app<br/>
| |- go.html # classification result page of web app<br/>
|- run.py # Flask file that runs app<br/>
data<br/>
|- disaster_categories.csv # data to process<br/>
|- disaster_messages.csv # data to process<br/>
|- process_data.py # python file to clean and output the data<br/>
|- disaster_categories.db # database to save clean data to<br/>
|- ETL Pipeline Preparation.ipynb # python notebook for EDA<br/>
models<br/>
|- train_classifier.py # python file to train and save the classifier<br/>
|- classifier.pkl # saved model<br/>
|- ML Pipeline Preparation.ipynb # python notebook for exploring model building<br/>
README.md

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

1. Run the following commands to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run the web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/

The following figures show the screenshots of the web.
![plot](images/disaster-response-project1.png)
![plot](images/disaster-response-project2.png)

<a name="license"></a>
## License
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
