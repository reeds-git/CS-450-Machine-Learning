from sklearn import datasets
import pandas as pd

def get_file_type():
    """
    Prompts to run Car, Iris, or Breast Cancer data
    :return: 1, 2, or 3 depending on what the user chooses
    """
    file_type = 0
    while file_type <= 0 or file_type >= 4:
        print("Enter the number of your file: \n\t1 Car\n\t2 Iris\n\t3 Breast Cancer")

        file_type = int(input())

    return file_type


def split_data(data_set):
    """
    Splits a data set into data and targets
    :param data_set: set of data to be separated into data and targets
    :return: data and target
    """
    return data_set.data, data_set.target


def load_file(file_type):
    """
    Load_file takes in a number to indicate the file type and gets the data and targets from the file from the data set
    :param file_type: file type by number. '1' is for Cars, '2' is for Iris, and '3' is for breast cancer
    :return: data and targets
    """
    header = None
    if file_type == 1:
        data_set = pd.read_csv('car.csv')

        data = data_set.ix[:, data_set.columns != "class"]
        targets = data_set.ix[:, data_set.columns == "class"]

        header = data_set.columns[:-1]

        return data.values, targets.values, header

    elif file_type == 2:

        data, targets = split_data(datasets.load_iris())

    elif file_type == 3:
        data, targets = split_data(datasets.load_breast_cancer())

    return data, targets, header
