from tensorflow import keras
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.datasets import mnist
from tensorflow.keras.regularizers import l1

import numpy as np

import matplotlib.pyplot as plt

import copy


class DenseAutoencoder():

    def __init__(self, input_shape, latent_dim, layer_sizes, activation="relu", regularized=True):
        self.input_shape = input_shape
        self.layer_sizes = layer_sizes
        self.activation = activation
        self.regularized = regularized
        self.latent_dim = latent_dim

        self.encoder = None
        self.decoder = None
        self.autoencoder = self._create_model()




    def fit(self, X, epochs, batch_size, shuffle=True):
        self.autoencoder.fit(X, X, epochs=epochs, batch_size=batch_size, shuffle=shuffle)
        print(type(self.autoencoder))
        

        # Creating encoder: 
        input = Input(shape=self.input_shape)
        encoded = input
        for layer in self.autoencoder.layers[:len(self.layer_sizes)+2]:
            encoded = layer(encoded)

        self.encoder = Model(input, encoded)
        self.encoder.summary()

        assert self.encoder.get_weights()[0].all() == self.autoencoder.get_weights()[0].all()

        # Creating decoder: 
        input = Input(shape=(self.latent_dim, ))
        decoded = input
        for layer in self.autoencoder.layers[len(self.layer_sizes)+2:]:
            decoded = layer(decoded)

        self.decoder = Model(input, decoded)
        self.decoder.summary()

        assert self.decoder.get_weights()[-1].all() == self.autoencoder.get_weights()[-1].all()



    def predict(self, X):
        predictions = self.autoencoder.predict(X)
        return predictions

    def encode(self, X):
        encoded = self.encoder.predict(X)
        return encoded 

    def decode(self, encoded):
        decoded = self.decoder.predict(encoded) 
        return decoded


    def _create_model(self):

        input = Input(shape=self.input_shape)

        encoded = input

        for size in self.layer_sizes:
            encoded = Dense(size, activation=self.activation)(encoded)
    
        encoded = Dense(self.latent_dim, activation="relu")(encoded)


        decoded = encoded
        for size in reversed(self.layer_sizes[:-1]):
            decoded = Dense(size, activation=self.activation)(decoded)

        decoded = Dense(self.input_shape[0], activation="sigmoid")(decoded)

        autoencoder = Model(input, decoded)
        autoencoder.compile(optimizer="adam", loss="binary_crossentropy")
        autoencoder.summary()

        return autoencoder