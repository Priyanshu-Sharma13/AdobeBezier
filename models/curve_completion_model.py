# from tensorflow.keras.layers import Input, Dense, Reshape, Flatten, Conv2DTranspose, Conv2D
# from tensorflow.keras.models import Sequential, Model
# from tensorflow.keras.optimizers import Adam

# def build_generator(latent_dim):
#     model = Sequential([
#         Dense(128, activation='relu', input_dim=latent_dim),
#         Reshape((8, 8, 2)),
#         Conv2DTranspose(64, (4, 4), activation='relu', padding='same'),
#         Conv2DTranspose(1, (7, 7), activation='sigmoid', padding='same')
#     ])
#     return model

# def build_discriminator():
#     model = Sequential([
#         Conv2D(64, (5, 5), padding='same', input_shape=(16, 16, 1)),
#         Flatten(),
#         Dense(1, activation='sigmoid')
#     ])
#     model.compile(loss='binary_crossentropy', optimizer=Adam(0.0002, 0.5))
#     return model

# # def build_gan(generator, discriminator):
# #     discriminator.trainable = False
# #     gan_input = Input(shape=(latent_dim,))
# #     x = generator(gan_input)
# #     gan_output = discriminator(x)
# #     gan = Model(gan_input, gan_output)
# #     gan.compile(loss='binary_crossentropy', optimizer=Adam(0.0002, 0.5))
# #     return gan

# def build_gan(generator, discriminator, latent_dim):
#     # Define the input for the GAN
#     gan_input = Input(shape=(latent_dim,))
    
#     # Generate an image from the latent space
#     x = generator(gan_input)
    
#     # Discriminate the generated image
#     gan_output = discriminator(x)
    
#     # Create the GAN model
#     gan = Model(inputs=gan_input, outputs=gan_output)
#     gan.compile(loss='binary_crossentropy', optimizer='adam')
    
#     return gan

import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Conv2D, Flatten, Reshape, BatchNormalization, LeakyReLU, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Reshape, LeakyReLU, BatchNormalization, Conv2DTranspose
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Flatten, Dropout, Dense, LeakyReLU
# from tensorflow.keras.models import Sequential
# from keras.models import Sequential
from keras.layers import Dense, Reshape, BatchNormalization, LeakyReLU, Conv2DTranspose, Input
from keras.optimizers import Adam


# def build_generator(latent_dim):
#     model = tf.keras.Sequential()
#     model.add(Dense(128 * 16 * 16, activation="relu", input_dim=latent_dim))
#     model.add(Reshape((16, 16, 128)))
#     model.add(Conv2D(128, kernel_size=3, padding="same"))
#     model.add(BatchNormalization())
#     model.add(LeakyReLU(alpha=0.2))
#     model.add(Conv2D(64, kernel_size=3, padding="same"))
#     model.add(BatchNormalization())
#     model.add(LeakyReLU(alpha=0.2))
#     model.add(Conv2D(1, kernel_size=3, padding="same", activation='tanh'))
#     return model

# def build_discriminator(input_shape=(16, 16, 1)):
#     model = tf.keras.Sequential()
#     model.add(Conv2D(64, kernel_size=3, strides=2, padding="same", input_shape=input_shape))
#     model.add(LeakyReLU(alpha=0.2))
#     model.add(Dropout(0.25))
#     model.add(Conv2D(128, kernel_size=3, strides=2, padding="same"))
#     model.add(LeakyReLU(alpha=0.2))
#     model.add(Dropout(0.25))
#     model.add(Flatten())
#     model.add(Dense(1, activation='sigmoid'))
#     return model




# def build_generator(latent_dim):
#     model = Sequential()
#     model.add(Dense(256, input_dim=latent_dim))
#     model.add(LeakyReLU(alpha=0.2))
#     model.add(BatchNormalization())
#     model.add(Reshape((8, 8, 32)))
#     model.add(Conv2DTranspose(64, kernel_size=4, strides=2, padding='same'))
#     model.add(LeakyReLU(alpha=0.2))
#     model.add(Conv2DTranspose(1, kernel_size=4, strides=2, padding='same', activation='tanh'))
#     model.summary()
#     return model


from keras.models import Sequential
from keras.layers import Dense, Reshape, BatchNormalization, LeakyReLU, Conv2DTranspose, Input

def build_generator(latent_dim=100):
    model = Sequential()

    # Input layer
    model.add(Input(shape=(latent_dim,)))

    # Dense layer
    model.add(Dense(4 * 4 * 256))  # Adjust to match the target size
    model.add(LeakyReLU(alpha=0.2))
    model.add(Reshape((4, 4, 256)))  # Reshape to match the Dense layer output

    # Deconvolutional layers
    model.add(Conv2DTranspose(128, kernel_size=3, strides=2, padding='same'))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.2))

    model.add(Conv2DTranspose(64, kernel_size=3, strides=2, padding='same'))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.2))

    model.add(Conv2DTranspose(1, kernel_size=3, strides=1, padding='same', activation='tanh'))

    model.summary()
    return model


# def build_generator(latent_dim=100):
#     model = Sequential()

#     # Input layer
#     model.add(Input(shape=(latent_dim,)))

#     # Dense layer
#     model.add(Dense(8 * 8 * 256))  # Adjust this size if necessary
#     model.add(LeakyReLU(alpha=0.2))
#     model.add(Reshape((8, 8, 256)))  # Match this with Dense output

#     # Deconvolutional layers
#     model.add(Conv2DTranspose(128, kernel_size=3, strides=2, padding='same'))
#     model.add(BatchNormalization())
#     model.add(LeakyReLU(alpha=0.2))

#     model.add(Conv2DTranspose(64, kernel_size=3, strides=2, padding='same'))
#     model.add(BatchNormalization())
#     model.add(LeakyReLU(alpha=0.2))

#     model.add(Conv2DTranspose(1, kernel_size=3, strides=1, padding='same', activation='tanh'))

#     model.summary()
#     return model


def build_discriminator(input_shape=(16, 16, 1)):
    model = Sequential()
    model.add(Conv2D(64, kernel_size=3, strides=2, padding='same', input_shape=input_shape))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.25))
    model.add(Conv2D(128, kernel_size=3, strides=2, padding='same'))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(1, activation='sigmoid'))
    model.summary()
    return model

def build_gan(generator, discriminator, latent_dim):
    discriminator.trainable = False
    gan_input = Input(shape=(latent_dim,))
    x = generator(gan_input)
    gan_output = discriminator(x)
    gan = Model(gan_input, gan_output)
    return gan