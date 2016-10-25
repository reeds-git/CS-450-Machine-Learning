import numpy as np
import ReadFile
from random import triangular as ran_float
from scipy.special import expit as sigmoid
from sklearn.cross_validation import train_test_split as split
from sklearn import preprocessing as prep


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
    :return: returns the number of neurons desired + 1 for the bias
    """
    return int(input("Enter the number of neurons you desire to have: "))


def get_num_layers():
    """
    Prompt the user for the number of layers to add to the network
    :return: returns the number of layers desired
    """
    return int(input("Enter the number of layers you desire to have: "))


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


class Neuron:
    """
    A Neuron contains weight, threshold=0, and bias=-1
    Contains a compute_g function that takes in and input and returns the h value.
    """
    def __init__(self, num_inputs):
        """
        Takes in an attribute and sets the weight to a random value between -2.0 & 2.0
        h_value = 0.0
        a_value = 0.0
        delta = 0.0
        Sets the threshold to 0.0
        Sets the bias to -1.0
        :param num_inputs: values of the attributes of the data
        """
        self.weight = [ran_float(-1.0, 1.0) for _ in range(num_inputs + 1)]
        self.h_value = 0.0
        self.a_value = 0.0
        self.delta = 0.0
        self.bias = -1.0

    def compute_a(self, input_values):
        """
        Add a bias to the list of inputs, compute the h for all of the inputs based on the weights,
          and returns a 1 if the neuron fired and a 0 if it didn't fire
        :param input_values: the attributes of the data
        :return: the a value
        """
        # add a bias, default is -1
        input_values = np.append(input_values, [self.bias])

        # h is the sum of each weight times the input
        self.h_value = sum([self.weight[w] * inputs for w, inputs in enumerate(input_values)])

        self.a_value = sigmoid(self.h_value)

        return self.a_value

    def compute_delta(self):

        return self.delta


def create_layer(num_inputs, num_neurons):
    """
    create the neurons each layer
    :param num_inputs: The number of attributes to add
    :param num_neurons: The number of neurons to create
    :return: A list of all of the neurons for that layer
    """
    return [Neuron(num_inputs) for _ in range(num_neurons)]


def forward_prop(data, network):
    """
    Takes a data set and a network of neurons
    Displays the "a" value of each neron on the layer
    :param data: the data to compute the a value
    :param network: a network of layers
    """
    for x in data:
        outputs = []
        for index, layer in enumerate(network):
            outputs.append([n.compute_a(outputs[index - 1] if index > 0 else x) for n in layer])
            print("The layer is {:d} has A values: ".format(index), outputs[index])
        print("---------------------")


def train_again():

    # get the file type
    file_type = ReadFile.get_file_type()

    # get the data set from a file and split into data and target
    data, targets, header = ReadFile.load_file(file_type)

    # Call functions to get the users input
    ts = float(get_test_size())
    num = int(get_random_state())
    num_layers = int(get_num_layers())

    # number of attributes or columns in the data set
    network = []
    num_targets = []

    for t in targets:
        if t not in num_targets:
            num_targets.append(t)

    # get the rest of the layers
    for x in range(num_layers + 1):

        if x == 0:
            num_inputs = data.shape[1]
            num_neurons = int(get_num_neurons())
        elif x == num_layers:
            num_inputs = len(network[x - 1])
            num_neurons = len(num_targets)
        else:
            num_inputs = len(network[x - 1])
            num_neurons = int(get_num_neurons())

        network.append(create_layer(num_inputs, num_neurons))

    # create 4 variables and splits the array into different parts
    training_data, test_data, training_target, test_target = split(data, targets, test_size=ts, random_state=num)

    # normalize the data
    std_train_data, std_test_data = ready_data(training_data, test_data)

    # compute all of the a values for each neuron and displays each a value of each neuron
    forward_prop(std_train_data, network)

    # check the accuracy
    #accuracy(test.train_knn(std_train_data, training_target, std_test_data), test_target)

    playing = input("\nDo you want to train again? (y or n)").lower()

    return playing

while train_again() == "y":
    pass
