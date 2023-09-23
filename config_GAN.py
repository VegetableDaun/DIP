# Path to data
path_to_data = 'chest-xray-pneumonia/chest_xray'

# Path to duplicated data
path_to_duplicate = 'duplicate.json'

# Path to the saved model
path_to_discriminator = 'Models/Discriminator'
path_to_generator = 'Models/Generator'

# Path to noise
path_to_noise = 'noise.json'

# Path to save metrics and image during train
path_to_result = 'train_models/cGAN'

# Count for test and valid
count_test = 1164
count_valid = 1164

# Add_labels is parameter for generator function that shows how count of images needs to increase.
# If your add_labels looks like {0: 0, 1: 1, 2: 2}
# this is meaning that image with label equal 1 will create one image with generator function.
add_labels = {0: 0, 1: 0, 2: 0}

# Setting for GAN models
batch_size = 32
num_channels = 3
num_classes = 3
image_size = 224
latent_dim = 128

generator_in_channels = latent_dim + num_classes
discriminator_in_channels = num_channels + num_classes