from sklearn.cross_validation import train_test_split as split
import KNN_Classifier
from HardCodedClassifier import HardCoded
import ReadFile as rf

file_type, file_name = rf.get_file_name()
iris = rf.load_file(file_type, file_name)

print(file_type, file_name)

# prompt user for test size and random state
def get_test_size():
    test_size = float(input("Enter a test size: "))

    while (test_size <= 0.0 or test_size > 0.5):
        test_size = float(input("Please enter a value between 0.0 and 0.5: "))

    return test_size

def get_random_state():
    random_state = int(input("Enter a random state size: "))

    while (random_state <= 0):
        random_state = int(input("Please enter a positive number."))

    return random_state

def accuracy(predicted_values, test_answers):
    num_predicted_values = 0

    for i in range(test_answers.size):
        num_predicted_values += predicted_values[i] == test_answers[i]

    print("The number of correct predictions: ", num_predicted_values)
    print("Total number tested:               ", test_answers.size)
    print("The percentage is {0:.2f}%".format((num_predicted_values / test_answers.size) * 100))

def train_again():

    # Call functions to get the users input
    ts = float(get_test_size())
    num = int(get_random_state())

    # create 4 variables and splits the array into different parts
    training_data, training_target, test_data, test_target = split(iris.data, iris.target, test_size=ts,
                                                                   random_state=num)
    # create an instance of the class
    test = HardCoded()
    test.train(training_data, training_target)
    accuracy(test.predict(test_data), test_target)

    playing = input("\nDo you want to train again? (y or n)").lower()

    return playing

while (train_again() == "y"):
    print("")