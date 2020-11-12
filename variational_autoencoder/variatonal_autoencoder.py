import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, Input, Conv2D, MaxPooling2D, UpSampling2D, Flatten, Reshape, Conv2DTranspose
from tensorflow.keras.datasets import mnist
from tensorflow.keras.regularizers import l1



class VariationalAutoencoder():

    def __init__(self, input_shape, latent_dim):
        self.input_shape = input_shape
        self.latent_dim = latent_dim


        self.encoder, self.decoder = self._create_model()

        self.vae = 


    def _sampling(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = keras.backend.random_normal(shape=(batch, dim))

        return z_mean + tf.exp(0.5 * z_log_var) * epsilon


    def _create_model(self):
        
        encoder_input = Input(shape=self.input_shape)
        
        encoded = encoder_input

        encoded = Conv2D(32, 3, activation='relu', strides=2, padding="same")(encoded)
        encoded = Conv2D(64, 3, activation='relu', strides=2, padding="same")(encoded)
        encoded = Flatten()(encoded)
        encoded = Dense(16)(encoded)

        z_mean = Dense(self.latent_dim, name="z_mean")(encoded)
        z_log_var = Dense(self.latent_dim, name="z_log_var")(encoded)

        z = self._sampling([z_mean, z_log_var])

        encoder = Model(encoder_input, [z_mean, z_log_var, z], name="encoder")
        encoder.summary() 

        decoder_input = Input(shape=(self.latent_dim, ))

        decoded = decoder_input
        decoded = Dense(7*7*64, activation='relu')(decoder_input)
        decoded = Reshape((7, 7, 64))(decoded)
        decoded = Conv2DTranspose(64, 3, activation='relu', strides=2, padding="same")(decoded)
        decoded = Conv2DTranspose(32, 3, activation='relu', strides=2, padding="same")(decoded)
        decoded = Conv2DTranspose(1, 3, activation='sigmoid', padding="same")(decoded)

        decoder = Model(decoder_input, decoded, name="decoder")
        decoder.summary()

        return encoder, decoder


if __name__ == "__main__":
    vae = VariationalAutoencoder((28, 28, 1), 2)