import tensorflow as tf


def initialize_model(input_size=256, output_size=512):
    model = tf.keras.Sequential()
    model.add(tf.keras.Input(shape=(input_size,)))
    model.add(tf.keras.layers.Dense(units=input_size, activation='relu'))
    model.add(tf.keras.layers.Dense(units=output_size, activation='relu'))
    model.build(input_shape=(input_size,))
    compile_model(model)
    return model


def compile_model(model: tf.keras.Model):
    model.compile(optimizer=tf.keras.optimizers.Adam(),
                  loss=tf.keras.losses.MeanSquaredError(),
                  metrics=tf.keras.metrics.CosineSimilarity())


def cosine_similarity(y_pred, y_true):
    cs = tf.keras.metrics.CosineSimilarity()
    cs.update_state(y_true, y_pred)
    return cs.result().numpy()


def import_pre_trained(model_path):
    return tf.keras.models.load_model(model_path)