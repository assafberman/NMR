import pandas as pd
import tensorflow as tf
from model import initialize_model
import numpy as np
#from database import import_database_as_df
from database_carbon import import_database_as_df
pd.set_option('display.max_columns', None)

nmr_df = import_database_as_df()
#train_test_cutoff = int(0.8*len(nmr_df['Input'].values))
#input_train = nmr_df['Input'].values[:train_test_cutoff]
#input_test = nmr_df['Input'].values[train_test_cutoff:]

model = initialize_model(input_size=180, output_size=512)
#model.summary()

#model.compile(optimizer='adam', loss=tf.keras.losses.CosineSimilarity(), metrics=['accuracy'])
