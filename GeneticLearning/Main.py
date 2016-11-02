import numpy as np
import ReadFile
from random import triangular as ran_float
from scipy.special import expit as sigmoid
from sklearn.cross_validation import train_test_split as split
from sklearn import preprocessing as prep
import matplotlib.pyplot as plot


def get_test_size():
    """
    prompt user for test size and random state (default is 0.3)
    :return: return a test size from 0.0 to 0.5
    """
    test_size = float(input("Enter a test size: ") or 0.3)

    while test_size <= 0.0 or test_size > 0.5:
        test_size = float(input("Please enter a value between 0.0 and 0.5: ") or 0.3)

    return test_size


def get_random_state():
    """
    prompt for a number to randomize the data (default is 12)
    :return: number
    """
    random_state = int(input("Enter a random state size: ") or 12)

    while random_state <= 0:
        random_state = int(input("Please enter a positive number.") or 12)

    return random_state


def get_num_neurons(current_layer):
    """
    Prompt the user for the number of neurons to add (default is 1)
    :return: returns the number of neurons desired + 1 for the bias
    """
    return int(input("Enter the number of neurons you desire for layer {}: ".format(current_layer)) or 1)


def get_num_layers():
    """
    Prompt the user for the number of layers to add to the network (default is 2)
    :return: returns the number of layers desired
    """
    return int(input("Enter the number of layers you desire to have: ") or 2)


def accuracy(predicted_values, test_answers):
    """
    Display the accuracy of the network
    :param predicted_values: A list of predicted values
    :param test_answers: A list of correct answers
    """
    num_predicted_values = 0

    for predict, correct in zip(predicted_values, test_answers):
        # index of the highest value
        if predict.tolist().index(max(predict)) == correct:
            num_predicted_values += 1

    print("The number of correct predictions: {}/{}".format(num_predicted_values, test_answers.size))
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
        a_value = 0.0
        delta = None
        Sets the bias to -1.0
        :param num_inputs: values of the attributes of the data
        """
        self.weights = [ran_float(-1.0, 1.0) for _ in range(num_inputs + 1)]
        self.a_value = 0.0
        self.bias = -1.0
        self.delta = None
        self.x = 0

    def compute_a(self, input_values):
        """
        Add a bias to the list of inputs, compute the h for all of the inputs based on the weights,
          and returns a 1 if the neuron fired and a 0 if it didn't fire
        :param input_values: the attributes of the data
        :return: the a value
        """
        # add a bias, default is -1
        input_values = np.append(input_values, [self.bias])

        self.x += 1
        print("Call compute a {} times".format(self.x))

        h_value = 0
        # h is the sum of each weight times the input
        for weight, inputs in enumerate(input_values):
            h_value += (self.weights[weight] * inputs)

        self.a_value = sigmoid(h_value)

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

    def create_layer(self, current_layer_num, data, targets, num_layers):
        """
        Create each layer of neurons in the network based upon the number of neurons specified
        :param current_layer_num:
        :param data: The data to add inputs for the input layer or layer 0
        :param targets: target values of the data to find the number of unique elements for the output nodes
        :param num_layers: Total number of layers to create
        :return:
        """
        num_targets = get_number_targets(targets)

        # create a hidden layer for the network
        if 0 < current_layer_num < (num_layers - 1):
            # create neurons with the inputs being the previous neurons activation value
            num_neurons = int(get_num_neurons(current_layer_num))
            return [Neuron(len(self.list_layers[current_layer_num - 1])) for _ in range(num_neurons)]

        # create an input layer for the network
        elif current_layer_num == 0:
            # create neurons with the inputs being the data
            num_neurons = int(get_num_neurons(current_layer_num))
            return [Neuron(data.shape[1]) for _ in range(num_neurons)]

        # create an output layer for the network
        else:
            # create neurons with the inputs being the previous neurons activation value
            return [Neuron(len(self.list_layers[current_layer_num - 1])) for _ in range(num_targets)]

    @staticmethod
    def compute_hidden_delta(a_value, list_weights, delta_right_layer):
        """
        Compute the delta value for the hidden layer neurons for back propagation
        :param a_value: The current neuron for the activation value
        :param list_weights: A list of all of the weights for that neuron
        :param delta_right_layer: The delta value from the neuron to the right of the current neuron
        :return: the computed delta value "a * (1 - a) * sum(weights * right_neuron_delta)"
        """
        return a_value * (1 - a_value) * sum([weight * delta for weight, delta in zip(list_weights, delta_right_layer)])

    @staticmethod
    def compute_output_delta(a_value, t_value):
        """
        Compute the delta value for the output layer neurons for back propagation
        :param a_value: The neuron for the activation value
        :param t_value: The target values from the data set
        :return: the computed delta value "a * (1 - a) * (a - target)"
        """
        return a_value * (1 - a_value) * (a_value - t_value)

    def compute_output_activation(self, inputs):
        """
        Calculate the activation values for all the neurons
        :param inputs: The inputs going into each neuron
        :return: A list of the activations
        """
        activations = []
        for pos, layer in enumerate(self.list_layers):
            activations.append([neuron.compute_a(activations[pos - 1] if pos > 0 else inputs) for neuron in layer])
        return activations

    def determine_correctness(self, targets, a_values):
        """
        Go backwards through the network and check for correctness
        :param targets: The target values
        :param a_values: List of all of the activations
        """
        for layer_pos, layer in reversed(list(enumerate(self.list_layers))):
            for neuron_pos, a_neuron in enumerate(layer):
                # get the hidden layers' deltas
                if layer_pos < (len(self.list_layers) - 1):
                    a_neuron.delta = self.compute_hidden_delta(
                        a_values[layer_pos][neuron_pos],
                        [neuron.weights[neuron_pos] for neuron in self.list_layers[layer_pos + 1]],
                        [neuron.delta for neuron in self.list_layers[layer_pos + 1]])

                # get the output layer's delta
                else:
                    a_neuron.delta = self.compute_output_delta(a_values[layer_pos][neuron_pos],
                                                               neuron_pos == targets)

    def update_all(self, row_data, a_value):
        """
        Go through each neuron in the layer and adjust weights according to the formula
        :param row_data: A single row of data
        :param a_value: the activation value to check to ensure correctness
        """
        for i, layer in enumerate(self.list_layers):
            for neuron in layer:
                self.update_weights(neuron, a_value[i - 1] if i > 0 else row_data.tolist())

    def update_weights(self, neuron, inputs):
        inputs = inputs + [-1]
        neuron.weights = [weight - self.learn_rate * neuron.delta * inputs[i] for i, weight in enumerate(neuron.weights)]

    def train_data(self, data, targets):
        """
        Train the network to predict the correct output
        :param data: Data set to train with
        :param targets: Correct outcomes that are to be predicted
        """
        # train for a set amount of times
        for _ in range(12):
            activation_list = []
            prediction = []
            for row_data, row_target in zip(data, targets):
                # calculate the activations of each neuron
                single_activation = self.compute_output_activation(row_data)
                prediction.append(self.compute_output_activation(row_data)[-1])

                # get the activations of the output layer of the network
                activation_list.append(single_activation)

                # find out which neurons' outputs were correct
                self.determine_correctness(targets, single_activation)

                # change the weights of all neurons that output the wrong answer
                self.update_all(row_data, single_activation)

    def predict(self, data):
        """
        Predict how many the neural network got correct
        :param data: A data set
        :return: predicted values
        """
        prediction = []
        for pos in data:
            prediction.append(self.compute_output_activation(pos)[-1])

        return prediction


def train_again():

    # get the file type
    file_type = ReadFile.get_file_type()

    # get the data set from a file and split into data and target
    data, targets, header = ReadFile.load_file(file_type)

    # create a network
    a_network = Network(data, targets)

    # Call functions to get the users input
    ts = float(get_test_size())
    num = int(get_random_state())

    # create 4 variables and splits the array into different parts
    training_data, test_data, training_target, test_target = split(data, targets, test_size=ts, random_state=num)

    # normalize the data
    std_train_data, std_test_data = ready_data(training_data, test_data)

    # train the data
    a_network.train_data(std_train_data, training_target)

    # check the accuracy
    accuracy(a_network.predict(std_test_data), test_target)

    return input("\nDo you want to train again? (y or n)").lower()

while train_again() == "y":
    pass
