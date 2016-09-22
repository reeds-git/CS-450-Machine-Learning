from sklearn import datasets
import pandas as pd


def get_file_name():
    print("Enter the number of your file type: \n\t1 CSV\n\t2 txt")

    file_type = input()

    file_name = input("Enter the file name: ")

    return (file_type, file_name)

# load the data
def load_file(file_type, file_name):
    data_set = datasets.load_iris()

    return data_set