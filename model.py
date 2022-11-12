import tensorflow as tf


def initialize_model(input_size=180, output_size=512):
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.InputLayer(input_shape=(input_size,)))
    model.add(tf.keras.layers.Dense(180, activation='relu'))
    model.add(tf.keras.layers.Dense(output_size, activation='relu'))
    model.build((input_size,))
    return model
