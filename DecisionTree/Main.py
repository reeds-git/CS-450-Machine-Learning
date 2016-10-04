import NodeClass as aNode
import numpy as np

def calc_entropy(self, var1, var2):
    entropy1 = (-var1 * np.log2(var1)) if var1 != 0 else 0
    entropy2 = (-var2 * np.log2(var2)) if var2 != 0 else 0

    total_entropy = entropy1 + entropy2

    print(total_entropy)

    return total_entropy, entropy1, entropy2


def average_ent(entropy_1, entropy_2, var1, var2):
    bob = (entropy_1 * var1) + (entropy_2 * var2)

    return bob

var1 = 2/3
var2 = 1/3

calc_entropy(var1, var2)