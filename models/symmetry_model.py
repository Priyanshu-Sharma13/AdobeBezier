# from tensorflow.keras.layers import Input, Dense, Lambda
# from tensorflow.keras.models import Model
# import tensorflow as tf
# import numpy as np

# def create_autoencoder(input_shape):
#     input_img = Input(shape=input_shape)
#     encoded = Dense(64, activation='relu')(input_img)
#     encoded = Dense(32, activation='relu')(encoded)
#     latent = Dense(16, activation='relu')(encoded)

#     decoded = Dense(32, activation='relu')(latent)
#     decoded = Dense(64, activation='relu')(decoded)
#     decoded = Dense(np.prod(input_shape), activation='sigmoid')(decoded)
#     decoded = Lambda(lambda x: tf.reshape(x, (-1, *input_shape)))(decoded)

#     autoencoder = Model(input_img, decoded)
#     autoencoder.compile(optimizer='adam', loss='binary_crossentropy')
#     return autoencoder


import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Lambda
from tensorflow.keras.models import Model
import numpy as np

def create_autoencoder(input_shape):
    inputs = Input(shape=input_shape)

    # Encoder
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2), padding='same')(x)

    # Bottleneck
    x = Conv2D(256, (3, 3), activation='relu', padding='same')(x)

    # Decoder
    x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)

    # Define output shape explicitly
    output_shape = input_shape

    # Use Lambda layer to reshape output
    decoded = Lambda(lambda x: tf.reshape(x, (-1, *output_shape)), output_shape=output_shape)(x)

    autoencoder = Model(inputs, decoded)
    autoencoder.compile(optimizer='adam', loss='binary_crossentropy')

    return autoencoder