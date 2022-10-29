from tensorflow.keras import models, layers


def initialize_embedding_model(input_spectrum, embedding_size=20, num_filters=4):
    input_spectrum = input_spectrum.reshape((input_spectrum.shape[0], input_spectrum.shape[1], 1))

    model = models.Sequential()
    # Adding zero at bottom of input matrix
    model.add(layer=layers.ZeroPadding2D(padding=((0, 1), (0, 0)), input_shape=input_spectrum.shape))
    # Convolving by (2, 2) kernel, thus reducing length and width of input matrix by 1
    model.add(layer=layers.Conv2D(filters=num_filters, kernel_size=(2, input_spectrum.shape[1]), strides=(1, 1),
                                  activation='relu'))
    if model.layers[-1].output_shape[1] % 2:
        model.add(layer=layers.Conv2D(filters=num_filters, kernel_size=(2, 1), strides=(1, 1), activation='relu'))
    while model.layers[-1].output_shape[1] != embedding_size:
        if model.layers[-1].output_shape[1] > embedding_size:
            model.add(layer=layers.MaxPooling2D(pool_size=(2, 1), strides=(2, 1)))
            model.add(layer=layers.Conv2D(filters=num_filters, kernel_size=(2, 1), strides=(1, 1), activation='relu',
                                          padding='same'))
            model.add(layer=layers.Conv2D(filters=num_filters, kernel_size=(3, 1), strides=(1, 1), activation='relu'))
        if model.layers[-1].output_shape[1] < embedding_size:
            model.add(layer=layers.UpSampling2D(size=(2, 1)))
            model.add(layer=layers.Conv2D(filters=num_filters, kernel_size=(2, 1), strides=(1, 1), activation='relu',
                                          padding='same'))
            #model.add(layer=layers.Conv2D(filters=num_filters, kernel_size=(3, 1), strides=(1, 1), activation='relu'))
        print(model.layers[-1].output_shape[1])

    model.build((1, input_spectrum.shape[0], input_spectrum.shape[1], 1))
    model.summary()
