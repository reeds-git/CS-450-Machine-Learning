from sklearn.cross_validation import train_test_split as split
from sklearn import preprocessing as prep
from KNN_Classifier import KNN
import ReadFile as rf


def get_test_size():
    """
    prompt user for test size and random state

    :return: return a test size from 0.0 to 0.5
    """
    test_size = float(input("Enter a test size: "))

    while test_size <= 0.0 or test_size > 0.5:
        test_size = float(input("Please enter a value between 0.0 and 0.5: "))

    return test_size


def get_random_state():
    """
    prompt for a number to randomize the data

    :return: number
    """
    random_state = int(input("Enter a random state size: "))

    while random_state <= 0:
        random_state = int(input("Please enter a positive number."))

    return random_state


def accuracy(predicted_values, test_answers):
    num_predicted_values = 0

    for i in range(test_answers.size):
        if predicted_values[i] == test_answers[i]:
            num_predicted_values += 1

    print("The number of correct predictions: ", num_predicted_values)
    print("Total number tested:               ", test_answers.size)
    print("The percentage is {0:.2f}%".format((num_predicted_values / test_answers.size) * 100))


def ready_data(training_data, testing_data):
    """
    Standardize the data for processing. See http://sebastianraschka.com/Articles/2014_about_feature_scaling.html
        for more info

    :param training_data: the data to be trained on
    :param testing_data: the data to test with
    :return: standardized training data and testing data
    """

    std_scale = prep.StandardScaler().fit(training_data)
    std_training_data = std_scale.transform(training_data)
    std_testing_data = std_scale.transform(testing_data)

    return std_training_data, std_testing_data


def train_again():

    # get the file type
    file_type = rf.get_file_type()

    # get the data set from a file and split into data and target
    data, targets = rf.load_file(file_type)

    # Call functions to get the users input
    ts = float(get_test_size())
    num = int(get_random_state())

    k = 5

    # create an instance of the class
    test = KNN()

    # create 4 variables and splits the array into different parts
    training_data, test_data, training_target, test_target = split(data, targets, test_size=ts, random_state=num)

    # normalize the data
    std_train_data, std_test_data = ready_data(training_data, test_data)

    #check the accuracy
    accuracy(test.train_knn(k, std_train_data, training_target, std_test_data), test_target)

    playing = input("\nDo you want to train again? (y or n)").lower()

    return playing


while train_again() == "y":
    print("")
