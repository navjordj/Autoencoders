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


    
if __name__ == "__main__":
    (x_train, _), (x_test, _) = mnist.load_data()
    x_train = x_train.astype('float32') / 255.
    x_test = x_test.astype('float32') / 255.
    x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
    x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))
    print(x_train.shape)
    print(x_test.shape)

    au = DenseAutoencoder(input_shape=(784,), layer_sizes=[128, 64, 32], latent_dim = 16)
    au.fit(x_train, epochs=100, batch_size=256)

    decoded = au.predict(x_test)

    n = 10  # How many digits we will display
    plt.figure(figsize=(20, 4))
    for i in range(n):
        # Display original
        ax = plt.subplot(2, n, i + 1)
        plt.imshow(x_test[i].reshape(28, 28))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        # Display reconstruction
        ax = plt.subplot(2, n, i + 1 + n)
        plt.imshow(decoded_imgs[i].reshape(28, 28))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    plt.show()

