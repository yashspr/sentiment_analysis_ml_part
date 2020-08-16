# How to Run

## Installation

> Note: Make sure sentiment_analysis_ml_part and web_sentiment_analysis are in a single root directory.

### Python Server

> Note: Make sure you have installed Microsoft C++ Build Tools before proceeding. 

1. Install anaconda
2. In terminal, navigate to sentiment_analysis_ml_part directory in anaconda part.
3. Run `conda env create -n sentiment_analysis -f ./environment.yml`
4. Activate the environment by running `conda activate sentiment_analysis`
5. Run this command `python -m spacy download en_core_web_sm`
6. Type in terminal `set FLASK_APP=server.py`
7. Then run `flask run`

### Nodejs Server

> Note: Make sure you have installed Nodejs and MongoDB before proceeding

1. Navigate to web_sentiment_analysis directory in CMD.
2. Type the command `npm install`

## Running The Project

### Python Server

1. Navigate to sentiment_analysis_ml_part directory in anaconda prompt.
2. Type in terminal `set FLASK_APP=server.py`
3. Then run `flask run`

### Nodejs Server

> Note: Make sure you have installed Nodejs and MongoDB before proceeding

1. Navigate to web_sentiment_analysis directory in CMD.
2. Type the command `npm run start`

The server will start. First time will take long because the models have to be trained and saved.