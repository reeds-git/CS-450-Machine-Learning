import numpy as np
import ReadFile
from random import triangular as ran_float
from scipy.special import expit as sigmoid
from sklearn.cross_validation import train_test_split as split
from sklearn import preprocessing as prep


def get_test_size():
    """
    prompt user for test size and random state (default is 0.3)
    :return: return a test size from 0.0 to 0.5
    """
    test_size = float(input("Enter a test size: "))

    while test_size <= 0.0 or test_size > 0.5:
        test_size = float(input("Please enter a value between 0.0 and 0.5: ") or 0.3)

    return test_size


def get_random_state():
    """
    prompt for a number to randomize the data (default is 12)
    :return: number
    """
    random_state = int(input("Enter a random state size: "))

    while random_state <= 0:
        random_state = int(input("Please enter a positive number.") or 12)

    return random_state


def get_num_neurons():
    """
    Prompt the user for the number of neurons to add (default is 1)
    :return: returns the number of neurons desired + 1 for the bias
    """
    return int(input("Enter the number of neurons you desire to have: ") or 1)


def get_num_layers():
    """
    Prompt the user for the number of layers to add to the network (default is 2)
    :return: returns the number of layers desired
    """
    return int(input("Enter the number of layers you desire to have: ") or 2)


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

        print("A-Value: ", self.a_value)
        return self.a_value


class Network:
    """
    Create a network with the number of neurons and layer specified by the user
    """
    def __init__(self, data, targets):
        """
        Create a network with the number of neurons and layer specified by the user
        :param data:
        :param targets:
        """
        self.list_layers = []
        self.learn_rate = .2
        self.num_layers = int(get_num_layers())  # total number of layers the user wants

        for layer in range(0, self.num_layers):
            self.list_layers.append(self.create_layer(layer, data, targets, self.num_layers))

    def create_layer(self, current_layer_num, data, targets, total_layers):
        """
        Create each layer of neurons in the network based upon the number of neurons specified
        :param current_layer_num:
        :param data: The data to add inputs for the input layer or layer 0
        :param targets: target values of the data to find the number of unique elements for the output nodes
        :param total_layers: Total number of layers to create
        :return:
        """
        num_targets = get_number_targets(targets)

        # create a hidden layer for the network
        if 0 < current_layer_num < (total_layers - 1):
            # create neurons with the inputs being the previous neurons activation value
            num_neurons = int(get_num_neurons())
            return [Neuron(len(self.list_layers[current_layer_num - 1])) for _ in range(num_neurons)]

        # create an input layer for the network
        elif current_layer_num == 0:
            # create neurons with the inputs being the data
            num_neurons = get_number_targets(targets)
            return [Neuron(data.shape[1]) for _ in range(num_neurons)]

        # create an output layer for the network
        else:
            # create neurons with the inputs being the previous neurons activation value
            return [Neuron(len(self.list_layers[current_layer_num - 1])) for _ in range(num_targets)]

    def compute_hidden_delta(self, a_neuron, list_weights, delta_right_layer):
        """
        Compute the delta value for the hidden layer neurons for back propagation
        a_value = activation value
        :param list_weights: A list of all of the weights for that neuron
        :param delta_right_layer: The delta value from the neuron to the right of the current neuron
        :return: the computed delta value "a * (1 - a) * sum(weights * right_neuron_delta)"
        """
        return a_neuron.a_value * (1 - a_neuron.a_value) * sum([weight * delta for weight, delta
                                                        in zip(list_weights, delta_right_layer)])

    def compute_output_delta(self, a_neuron, t_value):
        """
        Compute the delta value for the output layer neurons for back propagation
        :param a_neuron: The neuron for the activation value
        :param t_value: The target values from the data set
        :return: the computed delta value "a * (1 - a) * (a - target)"
        """
        return a_neuron.a_value * (1 - a_neuron.a_value) * (a_neuron.a_value - t_value)

    def create_new_weights(self, current_weights, delta_value, a_value):
        """
        Computes the new weight for each neuron
        :param current_weights: a list of current weights of each neuron
        :param delta_value: computed delta value based on layer
        :param a_value: activation value of the neuron
        """
        return current_weights - (self.learn_rate * delta_value * a_value)

    def forward_prop(self, data, network):
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
    #(std_train_data, test_target)

    # activation_list = forward_prop(std_train_data, network)

   # print("The output activations are:\n", activation_list)

    # check the accuracy
    #accuracy(std_train_data, test_target)

    return input("\nDo you want to train again? (y or n)").lower()

while train_again() == "y":
    pass
