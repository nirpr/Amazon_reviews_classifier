import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix
from sklearn.feature_selection import SelectKBest
import numpy as np


def load_json_data(file_path):
    """
    The function gets a file path, opens it and organizes it in a list of dictionaries.
    :param file_path: path to a Json file with multiple Json objects.
    :return: list of dictionaries.
    """
    json_list = []
    with open(file_path, 'r') as file:
        for jsonObj in file:
            json_dict = json.loads(jsonObj)
            json_list.append(json_dict)

    return json_list


def init_data(train_data, test_data):
    """
    Initializes the data from the list of dictionaries to train set and test set.
    :param train_data: list of dictionaries that represent multiple Json objects
    :param test_data: list of dictionaries that represent multiple Json objects
    :return: x and y train and test sets.
    """
    x_train = []
    y_train = []
    x_test = []
    y_test = []

    fields = ['reviewText', 'overall', 'summary', 'verified']
    for review in train_data:
        if all(field in review for field in fields) and review['verified']:
            if review['reviewText'] is not None and review['summary'] is not None:
                x_train.append(review['reviewText'] + '. ' + review['summary'])
                y_train.append(review['overall'])

    for review in test_data:
        if all(field in review for field in fields) and review['verified']:
            if review['reviewText'] is not None and review['summary'] is not None:
                x_test.append(review['reviewText'] + '. ' + review['summary'])
                y_test.append(review['overall'])

    return x_train, y_train, x_test, y_test


def print_k_best(x_train_vec, y_train, features_names):
    """
    The function finds and prints the 15 best features of the model.
    :param x_train_vec: a vector with x train data
    :param y_train: a list with y train data
    :param features_names: an array with features names
    :return: None
    """
    k_best = SelectKBest(k=15)
    k_best.fit_transform(x_train_vec, y_train)
    best_features = k_best.get_support()
    features = np.array(features_names)

    print(f'15 best features: {features[best_features]}')


def classify(train_file, test_file):
    """
    The function classify reviews to 5 different classes from 1 to 5, prints the confusion matrix
     and returns the accuracy of the test.
    :param train_file: path to a Json file with Json multiple objects
    :param test_file: path to a Json file with Json multiple objects
    :return: results of the test
    """
    print(f'starting feature extraction and classification, train data: {train_file} and test: {test_file}')
    train_data = load_json_data(train_file)
    test_data = load_json_data(test_file)

    x_train, y_train, x_test, y_test = init_data(train_data, test_data)

    ngram_range = (1, 1)
    vectorizer = TfidfVectorizer(ngram_range=ngram_range, max_features=1000)
    x_train_vec = vectorizer.fit_transform(x_train)
    x_test_vec = vectorizer.transform(x_test)
    features_names = vectorizer.get_feature_names_out()

    print_k_best(x_train_vec, y_train, features_names)

    classifier = LogisticRegression(max_iter=1000, solver='sag')
    classifier.fit(x_train_vec, y_train)

    y_pred = classifier.predict(x_test_vec)

    precision, recall, f1_score, _ = precision_recall_fscore_support(y_test, y_pred, average=None)

    # Compute accuracy
    accuracy = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    # Fill in the dictionary with actual scores obtained on the test data
    test_results = {'class_1_F1': f1_score[0],
                    'class_2_F1': f1_score[1],
                    'class_3_F1': f1_score[2],
                    'class_4_F1': f1_score[3],
                    'class_5_F1': f1_score[4],
                    'accuracy': accuracy}

    print(cm)
    return test_results


if __name__ == '__main__':
    with open('config.json', 'r') as json_file:
        config = json.load(json_file)

    results = classify(config['train_data'], config['test_data'])

    for k, v in results.items():
        print(k, v)
