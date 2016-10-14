import numpy as np
import ReadFile as rf
from random import triangular as ran_float
from sklearn.cross_validation import train_test_split as split
from sklearn import preprocessing as prep


class Neuron:
    def __init__(self, attributes):
        self.weight = [ran_float(-2.0, 2.0) for _ in range(attributes + 1)]
        self.threshold = 0.0
        self.bias = -1.0

    def compute_g(self, input_values):
        """
        Add a bias to the list of inputs, compute the h for all of the inputs based on the weights,
          and returns a 1 if the neuron fired and a 0 if it didn't fire
        :param inputs: the attributes of the data
        :return: 1 or 0 depending on if the neuron fires
        """
        # add a bias, default is -1
        input_values = np.append(input_values, self.bias)

        # h is the sum of each weight times the input
        compute_h = sum([self.weight[a_weight] * input for a_weight, input in enumerate(input_values)])

        # the threshold determines if the neuron fires
        if compute_h >= self.threshold:
            # neuron fires
            return 1
        else:
            # neuron doesn't fire
            return 0


def get_test_size():
    """
    prompt user for test size and random state
    :return: return a test size from 0.0 to 0.5
    """
    test_size = float(input("Enter a test size: "))

    while test_size <= 0.0 or test_size > 0.5:
        test_size = float(input("Please enter a value between 0.0 and 0.5: "))

    return test_size


def get_num_neurons():
    """
    Prompt the user for the number of neurons to add
    :return: returns the number of neurons desired
    """
    return int(input("Enter the number of neurons you desire to have: "))


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
    print("/", test_answers.size)
    print("The percentage is {0:.2f}%".format((num_predicted_values / test_answers.size) * 100))


def create_layer(num_neurons, num_attributes):
    """
    create the neurons each layer
    :param num_neurons: The number of neurons to create
    :param num_attributes: The number of attributes to add
    :return: A list of all of the neurons
    """
    return [Neuron(num_attributes) for _ in range(num_neurons)]


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
    data, targets, header = rf.load_file(file_type)

    # Call functions to get the users input
    ts = float(get_test_size())
    num = int(get_random_state())
    num_neurons = int(get_num_neurons())

    # create 4 variables and splits the array into different parts
    training_data, test_data, training_target, test_target = split(data, targets, test_size=ts, random_state=num)

    # normalize the data
    std_train_data, std_test_data = ready_data(training_data, test_data)

    # number of attributes or columns in the data set
    num_attributes = data.shape[1]

    # a list of each neuron a layer
    layer = create_layer(num_neurons, num_attributes)

    for x in data:
        outputs = [n.compute_g(x) for n in layer]
        print("The layer is : ", outputs)

    # check the accuracy
    # accuracy(test.train_knn(std_train_data, training_target, std_test_data), test_target)

    playing = input("\nDo you want to train again? (y or n)").lower()

    return playing

while train_again() == "y":
    pass
