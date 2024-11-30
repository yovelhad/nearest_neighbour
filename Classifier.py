import numpy as np
from typing import Tuple


class Classifier:

    k: int
    #input_to_labels: dict[Tuple[np.array]: int]

    def __init__(self, k: int, input_to_labels):
        self.k = k
        self.input_to_labels = input_to_labels

