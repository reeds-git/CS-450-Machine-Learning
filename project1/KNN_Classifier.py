from sklearn.cross_validation import train_test_split as split
import numpy as np
import ReadFile

class KNN:
    def knn(self, K, data, dataClass, inputs):
        num_inputs = np.shape(inputs)[0]
        closest_neighbor = np.zeros(num_inputs)

        for n in range(num_inputs):
            # Compute distances
            distances = np.sum((data-inputs[n,:])**2, axis=1)

            # Identify the nearest neighbors
            indices = np.argsort(distances, axis=0)

            classes = np.unique(dataClass[indices[:K]])
            if len(classes)==1:
                closest_neighbor[n] = np.unique(classes)

            else:
                counts = np.zeros(max(classes)+1)
                for i in range(K):
                    counts[dataClass[indices[i]]] += 1

                closest_neighbor[n] = np.max(counts)

        return closest_neighbor