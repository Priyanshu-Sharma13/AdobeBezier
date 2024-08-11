# # import numpy as np
# # from models.shape_model import create_shape_model
# # from models.symmetry_model import create_autoencoder
# # from models.curve_completion_model import build_generator, build_discriminator, build_gan
# # from utils.data_preprocessing import read_csv
# # from utils.feature_extraction import extract_features
# # from utils.visualization import plot_predictions

# # def main():
# #     # Load and preprocess data
# #     data = read_csv('data/input/frag0.csv')
# #     features = extract_features(data)

# #     # Define latent_dim
# #     latent_dim = 100  # Adjust as needed

# #     # Create and train models

# #     # shape_model = create_shape_model(input_shape=(64, 64, 1))
# #     # symmetry_model = create_autoencoder(input_shape=(64, 64, 1))
# #     # generator = build_generator(latent_dim=100)
# #     # discriminator = build_discriminator()
# #     # gan = build_gan(generator, discriminator)
    
# #     shape_model = create_shape_model(input_shape=(64, 64, 1))
# #     symmetry_model = create_autoencoder(input_shape=(64, 64, 1))
# #     generator = build_generator(latent_dim=latent_dim)
# #     discriminator = build_discriminator()
# #     gan = build_gan(generator, discriminator, latent_dim=latent_dim)

# #     # Implement training code here

# #     # Visualize results
# #     plot_predictions(data)

# # if __name__ == '__main__':
# #     main()







# # #     import numpy as np
# # # from models.shape_model import create_shape_model
# # # from models.symmetry_model import create_autoencoder
# # # from models.curve_completion_model import build_generator, build_discriminator, build_gan
# # # from utils.data_preprocessing import read_csv
# # # from utils.feature_extraction import extract_features
# # # from utils.visualization import plot_predictions

# # # def main():
# # #     # Load and preprocess data
# # #     data = read_csv('data/input/frag0.csv')
# # #     features = extract_features(data)

# #     # Define latent_dim
# #     # latent_dim = 100  # Adjust as needed

# #     # Create and train models
# #     # shape_model = create_shape_model(input_shape=(64, 64, 1))
# #     # symmetry_model = create_autoencoder(input_shape=(64, 64, 1))
# #     # generator = build_generator(latent_dim=latent_dim)
# #     # discriminator = build_discriminator()
# #     # gan = build_gan(generator, discriminator, latent_dim=latent_dim)

# #     # Implement training code here
# #     # For example:
# #     # gan.fit(...)

# #     # Visualize results
# # #     plot_predictions(data)

# # # if _name_ == '_main_':
# # #     main()



# import numpy as np
# import matplotlib.pyplot as plt
# from tensorflow.keras.optimizers import Adam
# from sklearn.preprocessing import MinMaxScaler
# from models.shape_model import create_shape_model
# from models.symmetry_model import create_autoencoder
# from models.curve_completion_model import build_generator, build_discriminator, build_gan
# from utils.data_preprocessing import read_csv
# from utils.feature_extraction import extract_features
# from utils.visualization import plot_predictions

# # def train_gan(gan, generator, discriminator, data, epochs=10000, batch_size=64, latent_dim=100):
# #     # Define loss and optimizers
# #     gan.compile(loss='binary_crossentropy', optimizer='adam')
# #     discriminator.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# #     for epoch in range(epochs):
# #         # Train discriminator
# #         idx = np.random.randint(0, data.shape[0], batch_size)
# #         real_images = data[idx]
# #         fake_images = generator.predict(np.random.randn(batch_size, latent_dim))
        
# #         # Labels for real and fake images
# #         real_labels = np.ones((batch_size, 1))
# #         fake_labels = np.zeros((batch_size, 1))

# #         # Train discriminator on real images
# #         d_loss_real = discriminator.train_on_batch(real_images, real_labels)
        
# #         # Train discriminator on fake images
# #         d_loss_fake = discriminator.train_on_batch(fake_images, fake_labels)
        
# #         # Train GAN
# #         noise = np.random.randn(batch_size, latent_dim)
# #         valid_y = np.ones((batch_size, 1))
        
# #         g_loss = gan.train_on_batch(noise, valid_y)
        
# #         if epoch % 1000 == 0:
# #             print(f"{epoch} [D loss: {d_loss_real[0]} | D accuracy: {100*d_loss_real[1]}] [D loss fake: {d_loss_fake[0]} | D accuracy fake: {100*d_loss_fake[1]}] [G loss: {g_loss}]")


# # import numpy as np

# # def train_gan(gan, generator, discriminator, data, epochs=10000, batch_size=64, latent_dim=100):
# #     # Define loss and optimizers
# #     discriminator.compile(loss='binary_crossentropy', optimizer=Adam(0.0002, 0.5), metrics=['accuracy'])
# #     gan.compile(loss='binary_crossentropy', optimizer=Adam(0.0002, 0.5))

# #     # Normalize the data to the range [-1, 1] as typical GAN inputs
# #     scaler = MinMaxScaler(feature_range=(-1, 1))
# #     data = scaler.fit_transform(data)
# #     data = data.reshape(-1, 16, 16, 1)  # Adjust shape to match the discriminator input

# #     for epoch in range(epochs):
# #         # Train discriminator
# #         idx = np.random.randint(0, data.shape[0], batch_size)
# #         real_images = data[idx]
# #         fake_images = generator.predict(np.random.randn(batch_size, latent_dim))
        
# #         # Labels for real and fake images
# #         real_labels = np.ones((batch_size, 1))
# #         fake_labels = np.zeros((batch_size, 1))

# #         # Train discriminator on real images
# #         d_loss_real = discriminator.train_on_batch(real_images, real_labels)
        
# #         # Train discriminator on fake images
# #         d_loss_fake = discriminator.train_on_batch(fake_images, fake_labels)
        
# #         # Train GAN
# #         noise = np.random.randn(batch_size, latent_dim)
# #         valid_y = np.ones((batch_size, 1))
        
# #         g_loss = gan.train_on_batch(noise, valid_y)
        
# #         if epoch % 1000 == 0:
# #             print(f"{epoch} [D loss: {d_loss_real[0]} | D accuracy: {100*d_loss_real[1]}] [D loss fake: {d_loss_fake[0]} | D accuracy fake: {100*d_loss_fake[1]}] [G loss: {g_loss}]")


# # import numpy as np

# def normalize_data(data):
#     # Min-Max normalization to [-1, 1]
#     data_min = np.min(data)
#     data_max = np.max(data)
#     return 2 * ((data - data_min) / (data_max - data_min)) - 1


# # def train_gan(gan, generator, discriminator, data):
# #     # Determine the total number of elements and target shape
# #     total_elements = data.size
# #     target_shape_elements = 16 * 16 * 1

# #     # Check if the total size is a multiple of the target shape
# #     if total_elements % target_shape_elements != 0:
# #         num_samples = total_elements // target_shape_elements
# #         print(f"Warning: Dataset size is {total_elements}, truncating to fit into the target shape.")
# #         data = data[:num_samples * target_shape_elements]

# #     # Reshape the data
# #     data = data.reshape(-1, 16, 16, 1)
    
# #     print("Reshaped data shape:", data.shape)

# #     # Continue with the rest of your training code
# #     # ...


# def train_gan(gan, generator, discriminator, data, epochs=10000, batch_size=64, latent_dim=100):
#     total_elements = data.size
#     target_shape_elements = 16 * 16 * 1

#     # Check if the total size is a multiple of the target shape
#     if total_elements % target_shape_elements != 0:
#         num_samples = total_elements // target_shape_elements
#         print(f"Warning: Dataset size is {total_elements}, truncating to fit into the target shape.")
#         data = data[:num_samples * target_shape_elements]

#     # Reshape the data
#     data = data.reshape(-1, 16, 16, 1)
    
#     print("Reshaped data shape:", data.shape)
    
#     # Define loss and optimizers
#     discriminator.compile(loss='binary_crossentropy', optimizer=Adam(0.0002, 0.5), metrics=['accuracy'])
#     gan.compile(loss='binary_crossentropy', optimizer=Adam(0.0002, 0.5))

#     # Normalize the data to the range [-1, 1]
#     data = normalize_data(data)
#     data = data.reshape(-1, 16, 16, 1)  # Adjust shape to match the discriminator input

#     for epoch in range(epochs):
#         print("Original data shape:", data.shape)
#         # Train discriminator
#         idx = np.random.randint(0, data.shape[0], batch_size)
#         real_images = data[idx]
#         fake_images = generator.predict(np.random.randn(batch_size, latent_dim))
        
#         # Labels for real and fake images
#         real_labels = np.ones((batch_size, 1))
#         fake_labels = np.zeros((batch_size, 1))

#         # Train discriminator on real images
#         d_loss_real = discriminator.train_on_batch(real_images, real_labels)
        
#         # Train discriminator on fake images
#         d_loss_fake = discriminator.train_on_batch(fake_images, fake_labels)
        
#         # Train GAN
#         noise = np.random.randn(batch_size, latent_dim)
#         valid_y = np.ones((batch_size, 1))
        
#         g_loss = gan.train_on_batch(noise, valid_y)
        
#         if epoch % 1000 == 0:
#             print(f"{epoch} [D loss: {d_loss_real[0]} | D accuracy: {100*d_loss_real[1]}] [D loss fake: {d_loss_fake[0]} | D accuracy fake: {100*d_loss_fake[1]}] [G loss: {g_loss}]")


# def generate_and_show_image(generator, latent_dim):
#     # Generate random noise
#     noise = np.random.normal(0, 1, (1, latent_dim))  # latent_dim should match the input shape of your generator

#     # Generate an image
#     generated_image = generator.predict(noise)

#     # Reshape and display the image
#     generated_image = generated_image.reshape(16, 16)  # Adjust to the output shape of the generator
#     plt.imshow(generated_image, cmap='gray')
#     plt.show()


# def main():
#     # Load and preprocess data
#     data = read_csv('data/input/occlusion1.csv')
#     features = extract_features(data)

#     train_gan(gan, generator, discriminator, data)

#     # Define latent_dim
#     latent_dim = 100  # Adjust as needed

#     # latent_dim = 100  # Update if your latent_dim is different
#     noise = np.random.normal(0, 1, (1, latent_dim))  # Generate random noise
#     generated_image = generator.predict(noise)[0]  # Generate image
#     generated_image = (generated_image + 1) / 2.0  # Rescale from [-1, 1] to [0, 1]

#     # Save the generated image
#     plt.imshow(generated_image[:, :, 0], cmap='gray')  # Update for grayscale images
#     plt.axis('off')
#     plt.savefig('generated_image.png')
#     plt.show()


#     # Create and train models

#     shape_model = create_shape_model(input_shape=(64, 64, 1))
#     symmetry_model = create_autoencoder(input_shape=(64, 64, 1))
#     generator = build_generator(latent_dim=latent_dim)
#     discriminator = build_discriminator()
#     gan = build_gan(generator, discriminator, latent_dim=latent_dim)

#     # Train the GAN
   

#     # Visualize results
#     plot_predictions(data)



# # Assuming generator is your trained generator model and latent_dim is the noise dimension
#     generate_and_show_image(generator, latent_dim)

# if __name__ == '__main__':
#     main()


import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler
from models.shape_model import create_shape_model
from models.symmetry_model import create_autoencoder
from models.curve_completion_model import build_generator, build_discriminator, build_gan
from utils.data_preprocessing import read_csv
from utils.feature_extraction import extract_features
from utils.visualization import plot_predictions

def normalize_data(data):
    # Min-Max normalization to [-1, 1]
    data_min = np.min(data)
    data_max = np.max(data)
    return 2 * ((data - data_min) / (data_max - data_min)) - 1

def train_gan(gan, generator, discriminator, data, epochs=10000, batch_size=64, latent_dim=100):
    total_elements = data.size
    target_shape_elements = 16 * 16 * 1

    # Check if the total size is a multiple of the target shape
    if total_elements % target_shape_elements != 0:
        num_samples = total_elements // target_shape_elements
        print(f"Warning: Dataset size is {total_elements}, truncating to fit into the target shape.")
        data = data[:num_samples * target_shape_elements]

    # Reshape the data
    data = data.reshape(-1, 16, 16, 1)
    
    print("Reshaped data shape:", data.shape)
    
    # Define loss and optimizers
    discriminator.compile(loss='binary_crossentropy', optimizer=Adam(0.0002, 0.5), metrics=['accuracy'])
    gan.compile(loss='binary_crossentropy', optimizer=Adam(0.0002, 0.5))

    # Normalize the data to the range [-1, 1]
    data = normalize_data(data)

    for epoch in range(epochs):
        # Train discriminator
        idx = np.random.randint(0, data.shape[0], batch_size)
        real_images = data[idx]
        fake_images = generator.predict(np.random.randn(batch_size, latent_dim))
        
        # Labels for real and fake images
        real_labels = np.ones((batch_size, 1))
        fake_labels = np.zeros((batch_size, 1))

        # Train discriminator on real images
        d_loss_real = discriminator.train_on_batch(real_images, real_labels)
        
        # Train discriminator on fake images
        d_loss_fake = discriminator.train_on_batch(fake_images, fake_labels)
        
        # Train GAN
        noise = np.random.randn(batch_size, latent_dim)
        valid_y = np.ones((batch_size, 1))
        
        g_loss = gan.train_on_batch(noise, valid_y)
        
        if epoch % 1000 == 0:
            print(f"{epoch} [D loss: {d_loss_real[0]} | D accuracy: {100*d_loss_real[1]}] [D loss fake: {d_loss_fake[0]} | D accuracy fake: {100*d_loss_fake[1]}] [G loss: {g_loss}]")

def generate_and_show_image(generator, latent_dim):
    # Generate random noise
    noise = np.random.normal(0, 1, (1, latent_dim))  # latent_dim should match the input shape of your generator

    # Generate an image
    generated_image = generator.predict(noise)

    # Rescale from [-1, 1] to [0, 1]
    generated_image = (generated_image + 1) / 2.0

    # Reshape and display the image
    generated_image = generated_image.reshape(16, 16)
    plt.imshow(generated_image, cmap='gray')
    plt.axis('off')
    plt.savefig('generated_image.png')
    plt.show()

def main():
    # Load and preprocess data
    data = read_csv('data/input/frag0.csv')
    features = extract_features(data)

    # Define latent_dim
    latent_dim = 100

    # Create and train models
    shape_model = create_shape_model(input_shape=(64, 64, 1))
    symmetry_model = create_autoencoder(input_shape=(64, 64, 1))
    generator = build_generator(latent_dim=latent_dim)
    discriminator = build_discriminator()
    gan = build_gan(generator, discriminator, latent_dim=latent_dim)
    
    # Generate and save an image
    generate_and_show_image(generator, latent_dim)

    # Visualize results
    plot_predictions(data)

    # Train the GAN
    train_gan(gan, generator, discriminator, data)

    

    

if __name__ == "__main__":
    main()