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

    for i, x in zip(predicted_values, test_answers):
        # index of the highest value
        if i == x.index(max(x)):
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


def get_number_targets(targets):
    """
    Gets the number of unique target values
    :param targets: list of target values
    :return: number of unique targets
    """
    num_targets = []
    for t in targets:
        if t not in num_targets:
            num_targets.append(t)

    return len(num_targets)


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

        print("inputs {}\nWeights {}".format(input_values, self.weight))

        # h is the sum of each weight times the input
        self.h_value = sum([self.weight[w] * inputs for w, inputs in enumerate(input_values)])
        print("H = ", self.h_value)
        self.a_value = sigmoid(self.h_value)

        return self.a_value

    def compute_hidden_delta(self):
        return self.a_value * (1 - self.a_value)

    def compute_output_delta(self, t_value):
        return self.a_value * (1 - self.a_value) * (self.a_value - t_value)


def create_layer(num_inputs, num_neurons):
    """
    Create each neuron based upon the number of neurons specified
    :param num_inputs: How many inputs to create weights for
    :param num_neurons: The number of neurons to be created in each level
    :return: list of neurons with random weights set
    """
    layer = []
    for x in range(num_neurons):
        layer.append(Neuron(num_inputs))

    return layer


def create_network(data, targets):
    """
    Create a network with the number of neurons and layer specified by the user
    :param data: List of the data set values
    :param targets: List of target values of the data set
    :return: A list of neurons connected by layers to from a network
    """
    num_layers = int(get_num_layers())

    # list of a layer list of neurons
    network = []

    # get the rest of the layers
    for layer in range(num_layers + 1):
        if layer == 0:
            num_inputs = data.shape[1]
            num_neurons = int(get_num_neurons())
        elif layer == num_layers:
            num_inputs = len(network[layer - 1])
            num_neurons = get_number_targets(targets)
        else:
            num_inputs = len(network[layer - 1])
            num_neurons = int(get_num_neurons())

        network.append(create_layer(num_inputs, num_neurons))

    return network


def forward_prop(data, network):
    """
    Takes a data set and a network of neurons
    Displays the activation value of each neron on the layer
    :param data: the data to compute the a value
    :param network: a network of layers
    """
    a_values_list = []
    output_a_values = []

    for row in data:
        # for each layer in the network
        for num_layer, layer in enumerate(network):
            print("--------------- Layer {} ---------------------".format(num_layer))
            # compute the activation value for each neuron in the layer
            x = 0
            for neuron in layer:

                print("%%%%%%%% Neuron {}   %%%%%%%%%%%%%%%%%%%%%%%".format(x))
                x += 1
                if (num_layer - 1) > 0:
                    # Activation computed based on the previous activation value
                    print("\t\t\t\t\t\tThe a of previous layer: ", a_values_list[num_layer - 2])
                    a_values_list.append(neuron.compute_a(a_values_list[num_layer - 1]))
                    print("\t\t\t\t\t\tThe a neuron: ", a_values_list[num_layer])
                elif num_layer == 0:
                    # Activation computed based on data
                    print("\t\t\t\t\t\tLayer 0 A value: ", neuron.compute_a(row))
                    a_values_list.append(neuron.compute_a(row))
                else:
                    # Add the output layers activation values
                    output_a_values.append(neuron.compute_a(a_values_list[num_layer - 1]))

        print("\n\tmax a: ", output_a_values)
        print("\n********************************************************\n")

    return output_a_values


def train_again():

    # get the file type
    file_type = ReadFile.get_file_type()

    # get the data set from a file and split into data and target
    data, targets, header = ReadFile.load_file(file_type)

    # Call functions to get the users input
    ts = float(get_test_size())
    num = int(get_random_state())

    # create 4 variables and splits the array into different parts
    training_data, test_data, training_target, test_target = split(data, targets, test_size=ts, random_state=num)

    # normalize the data
    std_train_data, std_test_data = ready_data(training_data, test_data)

    # number of attributes or columns in the data set
    network = create_network(std_train_data, test_target)

    print(network)

    activation_list = forward_prop(std_train_data, network)

    print("The output activations are:\n", activation_list)

    # check the accuracy
    #accuracy(std_train_data, test_target)

    return input("\nDo you want to train again? (y or n)").lower()

while train_again() == "y":
    pass
