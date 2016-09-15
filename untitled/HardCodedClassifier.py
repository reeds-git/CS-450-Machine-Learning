from sklearn import datasets
from sklearn.cross_validation import train_test_split as split

# load the data
iris = datasets.load_iris()

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

class HardCoded:
    # define a variable
    def train(self, data_to_train, target_to_train):
        print("Lets train\n")

    #
    def predict(self, data_set_to_predict):
        # create a empty list
        val = []

        # fill with 0 to reset the values
        for i in data_set_to_predict:
            val.append(0)

        return val

def accuracy(predicted_values, test_answers):
    num_predicted_values = 0

    for i in range(test_answers.size):
        num_predicted_values += predicted_values[i] == test_answers[i]

    print("The number of correct predictions: ", num_predicted_values, "\nTotal number tested: ", test_answers.size)
    print("The percentage is {0:.2f}%".format((num_predicted_values / test_answers.size) * 100))

def play_again():
    # create an instance of the class

    # Call functions
    ts = float(get_test_size())
    num = int(get_random_state())

    # create 4 variables and splits the array into different parts
    training_data, training_target, test_data, test_target = split(iris.data, iris.target, test_size=ts,
                                                                   random_state=num)

    test = HardCoded()
    test.train(training_data, training_target)
    accuracy(test.predict(test_data), test_target)

    playing = input("\nDo you want to play again? (y or n)").lower()

    return playing

while (play_again() == "y"):
    print("")

""""**********************************************************************************************************"""
# show the data (the attributes of each instance)
print(iris.data)

# show the target values (in numeric format) of each instance
print(iris.target)

# show the actual target names that correspond to each number
print(iris.target_names)
