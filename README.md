# Amazon Review Classifier

## Description
This repository contains a Python program for classifying Amazon reviews into five classes of ratings using natural language processing techniques. The classifier employs the TF-IDF vectorization method in combination with logistic regression to achieve accurate classification results.

## Files
1. `classifier.py`: The main Python script that performs the classification task.
2. `config.json`: Configuration file specifying the file paths for training and testing data.
3. `data/`: Directory containing the JSON files for training and testing data.

## Installation
1. Clone this repository to your local machine.

## Usage
1. Update the `config.json` file with the correct file paths for your training and testing data.
2. Run the `main.py` script.


## Output
The program will output the classification results, including accuracy metrics and confusion matrix, for the test data.

## Dataset
The dataset used for this classification task consists of Amazon reviews. Each review is labeled with one of five rating classes.

## Approach
1. Data Preparation: The program reads the JSON files and organizes the data into a list of dictionaries.
2. Feature Extraction: TF-IDF vectorization is applied to convert text data into numerical features.
3. Model Training: Logistic regression model is trained on the TF-IDF vectors of the training data.
4. Model Evaluation: The trained model is evaluated on the testing data to measure its performance.

## License
This project is licensed under the MIT License. See the `LICENSE` file for details.
