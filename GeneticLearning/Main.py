import numpy as np
import ReadFile as rf
from scipy.special import expit as g
from random import triangular as ran_float
from sklearn.cross_validation import train_test_split as split
from sklearn import preprocessing as prep


class Neuron:
    """
    A Neuron contains weight, threshold=0, and bias=-1
    Contains a compute_g function that takes in and input and returns the h value.
    """

    def __init__(self, num_inputs):
        """
        Takes in an attribute and sets the weight to a random value between -1.0 & 1.0
        Sets the threshold to 0.0
        Sets the bias to -1.0
        :param num_inputs: the number of inputs into a neuron
        """
        self.weight = [0.1, 0.3, 0.5]
        '''  self.weight = []
        for _ in range(num_inputs + 1):
            self.weight.append(ran_float(-1.0, 1.0))
        '''
        print("The weight is ", self.weight)
        self.threshold = 0.0
        self.bias = -1.0

    def compute_a(self, input_values):
        """
        Add a bias to the list of inputs, compute the h for all of the inputs based on the weights,
        :param input_values: the attributes of the data
        :return: the activation value of a neuron
        """
        input_values = np.append(input_values, [self.bias])
        return g(sum([self.weight[a_weight] * x for a_weight, x in enumerate(input_values)]))


class Layer:
    def __init__(self):
        self.layer_num = 0.0
        self.num_neurons = 0
        self.exit_layer = False
        self.current_layer = []
        self.num_layers = 0

    def get_num_neurons(self):
        """
        Prompt the user for the number of neurons to add
        Sets the number of neurons desired
        """
        self.num_neurons = int(input("Enter the number of neurons you desire to have: "))

    def get_num_layers(self):
        """
        Prompt the user for the number of layers to add
        Sets the number of layers desired
        """
        self.num_layers = int(input("Enter the number of layers you desire to have: "))

    def create_layer(self, num_attributes):
        """
        create the neurons each layer
        :param num_attributes: The number of attributes to add
        :return: A list of all of the neurons
        """

        [Neuron(num_attributes) for _ in range(self.num_neurons)]


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


def train_again():

    # get the file type
    file_type = rf.get_file_type()

    # get the data set from a file and split into data and target
    data, targets, header = rf.load_file(file_type)

    # Call functions to get the users input
    ts = float(get_test_size())
    num = int(get_random_state())

    # create 4 variables and splits the array into different parts
    training_data, test_data, training_target, test_target = split(data, targets, test_size=ts, random_state=num)

    # normalize the data
    std_train_data, std_test_data = ready_data(training_data, test_data)

    # number of attributes or columns in the data set
    num_attributes = data.shape[1]

    n = Neuron(0)

    activation = n.compute_a(data)

    print("The activation is : ", activation)
    # a list of each neuron a layer
#    layer = Layer()
 #   layer.get_num_layers()
  #  layer.get_num_neurons()
   # layer.create_layer(num_attributes)


    """for x in data:
        outputs = [n.compute_g(x) for n in layer]
        print("The layer is : ", outputs)
        """
    # check the accuracy
    # accuracy(test.train_knn(std_train_data, training_target, std_test_data), test_target)

    playing = input("\nDo you want to train again? (y or n)").lower()

    return playing

while train_again() == "y":
    pass
