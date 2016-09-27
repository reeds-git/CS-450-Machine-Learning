import numpy as np

class KNN:

    def train_knn(self, k, data, data_class, inputs):
        """
        K Nearest Neighbor from the book

        :param k: how many neighbors to check
        :param data: the data to process
        :param data_class: the standardized instance of the class
        :param inputs: what to check
        :return: the closest neighbor
        """
        num_inputs = np.shape(inputs)[0]
        closest_neighbor = np.zeros(num_inputs)

        for n in range(num_inputs):
            # Compute distances
            distances = np.sum((data - inputs[n, :]) ** 2, axis=1)

            # Identify the nearest neighbors
            indices = np.argsort(distances, axis=0)

            classes = np.unique(data_class[indices[:k]])

            if len(classes) == 1:
                closest_neighbor[n] = np.unique(classes)

            else:
                counts = np.zeros(max(classes) + 1)
                for i in range(k):
                    counts[data_class[indices[i]]] += 1

                closest_neighbor[n] = np.max(counts)

        return closest_neighbor
