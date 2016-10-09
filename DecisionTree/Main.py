import numpy as np
import ReadFile as rf
from sklearn.cross_validation import train_test_split as split
from sklearn import preprocessing as prep
import NodeClass as aNode

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


def calculate_entropy(data, a_class, attribute):

    # get the number of rows in the data
    num_rows = len(data)

    # loop through the rows and fill an array with the attribute's values
    values = []
    for item in data:
        if item[attribute] not in values:
            values.append(item[attribute])

    # get an array filled with "0" to store the number of attributes and entropy values
    num_values_in_attribute, attribute_entropies = np.zeros(len(values))
    num_value, value_in_column, class_index = 0
    column_values, class_values = []

    # loop for each value e.g. good, avg, and low in credit score
    for value in values:

        # count the number of occurrences of each item in a column of the data
        for data_point in data:
            if data_point[attribute] == value:
                num_values_in_attribute[num_value] += 1
                column_values.append(a_class[value_in_column])

            value_in_column += 1

            # Fill an array of values in the class column
            for item_in_class in class_values:
                # if the item is not in the array yet add it
                if class_values.count(item_in_class) == 0:
                    class_values.append(item_in_class)

            # array containing the number of each value of the class ## this starts empty
            num_class_values, entropy = np.zeros(len(class_values))

            # Count the number of occurrences of each item in the column
            for occurrence in class_values:
                for col_val in column_values:
                    if col_val == occurrence:
                        num_class_values[class_index] += 1
                class_index += 1

            # get the fraction of each item in the class
            for i in range(len(class_values)):
                entropy[num_value] += calculate_entropy(float(num_class_values[i]) / sum(num_class_values))

            # get the weight of an entropy  ex: (4/13 * entropy)
            weight = num_values_in_attribute[num_value] / num_rows
            entropy[num_value] = (entropy[num_value] * (num_values_in_attribute[num_value] / weight))
            num_value += 1

        return sum(entropy)


















def train_again():

    # get the file type
    file_type = rf.get_file_type()

    # get the data set from a file and split into data and target
    data, targets, header = rf.load_file(file_type)

    # Call functions to get the users input
    ts = float(get_test_size())
    num = int(get_random_state())

    #############################################################################


    ###########################################################################


    def calc_entropy(var):
        return (-var * np.log2(var1) if var != 0 else 0)


    def average_ent(entropy_1, entropy_2, var1, var2):
        bob = (entropy_1 * var1) + (entropy_2 * var2)

        return bob

    var1 = 3/5
    var2 = 2/5
    var3 = 4/5

    total_entropy = calc_entropy(var1) + calc_entropy(var2) + calc_entropy(var3)
    print(total_entropy)

    # create an instance of the class
    #test = KNN()


    # create 4 variables and splits the array into different parts
    train_data, test_data, train_target, test_target = split(data, targets, test_size=ts, random_state=num)

    # discreetize the data

    #check the accuracy
    #accuracy(test.train_knn(k, std_train_data, training_target, std_test_data), test_target)

    #######################################################################
    playing = input("\nDo you want to train again? (y or n)").lower()

    return playing

while train_again() == "y":
    pass