from model import initialize_embedding_model
import numpy as np

input_spectrum = np.array([[0.1, 0.9, 1.1], [3, 1, 3]]).T

initialize_embedding_model(input_spectrum)