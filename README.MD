# Problem statement

The goal of this project is to develop a machine learning model using Python to analyze Danish house prices. This model will predict house prices based on location.

# Motivation

Understanding house price trends is crucial for homeowners, buyers, and real estate investors to make informed decisions. By leveraging AI and machine learning, we can create predictive models that provide insights into future price movements.

# Theoretical foundation

The project will utilize concepts from machine learning such as: supervised learning, unsupervised learning and large language models.

# Design

Data Collection: From: https://rkr.statistikbank.dk/statbank5a/SelectVarVal/Define.asp?MainTable=UDB020&PLanguage=0&PXSId=0&wsid=cflastupd.

Data Preprocessing: Clean and preprocess the data to handle missing values, outliers, and categorical variables using elbow method, z-scores and one-hot encoding.

Model Selection: Evaluate various model such linear regression, random forest, support vector regression, kmeans clustering and LLM's to determine the most suitable for predicting house prices.

Model Training and Evaluation: Train the selected model on the training dataset and evaluate its performance using appropriate metrics such as MSE, MAE and R2 score.

Web app where you can predict future houseprices.

# Code

Data_cleaning.ipynb: since i couldn't download all house price data all at once it came in six different files that i then combine.

Data_procerssing.ipynb: processing the data and getting it ready for the ai models.

Supervised Learning.ipynb, Unsuper_vised Learning.ipynb, LLM.ipynb: Code for the different models.

App folder contains all code for the web app.

# Artefacts

All data can be found in the /Data folder with subfolders for raw data, data for the llm and training and test data.

# Implementation instructions

Install necessary python libraries.

Run the code at the bottom of Supervised Learning.ipynb

Cd into App folder and run: python -m pip install flask.

To run web app: python -m flask run