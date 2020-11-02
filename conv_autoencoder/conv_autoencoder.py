from tensorflow import keras
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, Input, Conv2D, MaxPooling2D, UpSampling2D
from tensorflow.keras.datasets import mnist
from tensorflow.keras.regularizers import l1

import numpy as np


import copy


class ConvAutoencoder():

    def __init__(self, input_shape, filter_sizes, kernel_size, strides=1, pool_shape=(2, 2), skip_last_padding=True, activation="relu"):
        self.input_shape = input_shape
        self.activation = activation
        self.filter_sizes = filter_sizes
        self.kernel_size = kernel_size
        self.strides = strides
        self.pool_shape = pool_shape
        self.skip_last_padding = skip_last_padding

        self.encoder = None
        self.decoder = None
        self.autoencoder = self._create_model()


    def fit(self, X, epochs, batch_size, shuffle=True, callback=[]):
        self.autoencoder.fit(X, X, epochs=epochs, batch_size=batch_size, shuffle=True, callbacks=callback)
        
    def predict(self):
        pass 

    def encode(self):
        raise NotImplementedError() 

    def decode(self):
        raise NotImplementedError()

    def _create_model(self):
        
        input = Input(shape=self.input_shape)

        encoded = input
        for filter_size in self.filter_sizes:
            encoded = Conv2D(filter_size, self.kernel_size, self.strides, padding="same", activation=self.activation)(encoded)
            encoded = MaxPooling2D(self.pool_shape, padding="same")(encoded)

        decoded = encoded
        for filter_size in reversed(self.filter_sizes[1:]):
            decoded = Conv2D(filter_size, self.kernel_size, self.strides, padding="same", activation=self.activation)(decoded)
            decoded = UpSampling2D(self.pool_shape)(decoded)

        decoded = Conv2D(self.filter_sizes[0], self.kernel_size, self.strides, activation=self.activation)(decoded)
        decoded = UpSampling2D(self.pool_shape)(decoded)

        decoded = Conv2D(1, self.kernel_size, self.strides, padding="same", activation=self.activation)(decoded)

        model = Model(input, decoded)
        model.summary()

    def _conv_output_shape(self):
        pass

        


if __name__ == "__main__":
    model = ConvAutoencoder(input_shape=(28, 28, 1), filter_sizes=[16, 8, 8], kernel_size=(3, 3))
