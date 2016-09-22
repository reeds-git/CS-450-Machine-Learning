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