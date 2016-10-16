from sklearn import datasets
import pandas as pd
pd.options.mode.chained_assignment = None  # stop warnings from displaying


def get_file_type():
    """
    Prompts to run (1) Pima Indians Diabetes or (2) Iris data set
    :return: 1 or 2 depending on what the user chooses
    """
    file_type = 0
    while file_type <= 0 or file_type >= 4:
        print("Enter the number of your file: \n\t1 Pima Indians Diabetes\n\t2 Iris")

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
    :param file_type: file type by number. '1' is for diabetes and '2' is for Iris
    :return: data and targets
    """
    header = None
    if file_type == 1:
        data_set = pd.read_csv('pima-indians-diabetes.csv', dtype=float)  # set data as floats

        data = data_set.ix[:, data_set.columns != "class"]
        targets = data_set.ix[:, data_set.columns == "class"]

        header = data_set.columns[:-1]

        return data.values, targets.values, header

    elif file_type == 2:

        data, targets = split_data(datasets.load_iris())

    return data, targets, header
