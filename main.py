import pandas as pd
import tensorflow as tf
from model import initialize_model
import numpy as np
from database import import_database_as_df
pd.set_option('display.max_columns', None)

nmr_df = import_database_as_df()
print(nmr_df)

#input_spectrum = np.array([[0.1, 0.9, 1.1, 5.8, 9.4, 9.8, 9.7], [3, 1, 3, 1, 3, 1, 3]]).T
#model = initialize_model(input_spectrum, embedding_size=32, num_filters=2, fingerprint_size=1024)
#model.summary()

#model.compile(optimizer='adam', loss=tf.keras.losses.CosineSimilarity(), metrics=['accuracy'])
