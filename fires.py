import os
import cv2
from PIL import Image
import torch
import torchvision.transforms as transforms
from torch.autograd.variable import Variable
import torch.nn as nn
import torch.optim as optim
import time

# GAN
# geckkkoooooooooo
# genniessssss
class generator(nn.Module):
    def __init__(self, input_dim=100, output_dim=1, input_size=32):
        super(generator, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.input_size = input_size

        self.fc = nn.Sequential(
            nn.Linear(self.input_dim, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Linear(1024, 128 * (self.input_size // 4) * (self.input_size // 4)),
            nn.BatchNorm1d(128 * (self.input_size // 4) * (self.input_size // 4)),
            nn.ReLU(),
        )
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, self.output_dim, 4, 2, 1),
            nn.Tanh(),
        )
        utils.initialize_weights(self)

    def forward(self, input):
        x = self.fc(input)
        x = x.view(-1, 128, (self.input_size // 4), (self.input_size // 4))
        x = self.deconv(x)

        return x
# dinnnnnies
class discriminator(nn.Module):
    def __init__(self, input_dim=1, output_dim=1, input_size=32):
        super(discriminator, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.input_size = input_size

        self.conv = nn.Sequential(
            nn.Conv2d(self.input_dim, 64, 4, 2, 1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
        )
        self.fc = nn.Sequential(
            nn.Linear(128 * (self.input_size // 4) * (self.input_size // 4), 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(0.2),
            nn.Linear(1024, self.output_dim),
            nn.Sigmoid(),
        )
        utils.initialize_weights(self)

    def forward(self, input):
        x = self.conv(input)
        x = x.view(-1, 128 * (self.input_size // 4) * (self.input_size // 4))
        x = self.fc(x)

        return x

def train_GAN():
    # Implement the training of the GAN here
    pass


def noise(size):
    '''
    Generates a 1-d vector sampled random values
    '''
    n = Variable(torch.randn(size, 100))
    return n

def video_to_frames(video, path_output_dir):
    def video_to_frames(video, path_output_dir):
    if not os.path.isfile(video):
        print("File path {} does not exist bro gtfo. Exiting...".format(video))
        return
    os.makedirs(path_output_dir, exist_ok=True)
    vidcap = cv2.VideoCapture(video)
    if not vidcap.isOpened():
        print("Could not open video file {}. laterboners...".format(video))
        return
    count = 0
    while vidcap.isOpened():
        success, image = vidcap.read()
        if success:
            try:
                cv2.imwrite(os.path.join(path_output_dir, '%d.png') % count, image)
                count += 1
            except Exception as e:
                print("Error while writing the fuggin frame: {}".format(e))
                break
        else:
            break
    cv2.destroyAllWindows() """eeepp"""
    vidcap.release()

def images_to_vectors(images):
    return images.view(images.size(0), 3072)

def vectors_to_images(vectors):
    return vectors.view(vectors.size(0), 3, 32, 32)

def transform_images(image_directory):
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    image_paths = [os.path.join(image_directory, image) for image in os.listdir(image_directory) if image.endswith(".png")]
    images = [transform(Image.open(image_path)) for image_path in image_paths]
    return torch.stack(images)
# where the stuff is at
video_directory = '/home/darkstar/Desktop/videos/'
base_directory = '/home/darkstar/Desktop/output/'
output_directory = '/home/darkstar/Desktop/output_images/'

video_paths = [os.path.join(video_directory, video) for video in os.listdir(video_directory) if video.endswith(".mp4")]

generator = Generator()
discriminator = Discriminator()

# Binary cross entropy: http://pytorch.org/docs/nn.html#bceloss
loss = nn.BCELoss()

# Optimizers
g_optimizer = optim.Adam(generator.parameters(), lr=0.0002)
d_optimizer = optim.Adam(discriminator.parameters(), lr=0.0002)

def ones_target(size):
    '''
    Tensor containing ones, with shape = size
    '''
    data = Variable(torch.ones(size, 1))
    return data

def zeros_target(size):
    '''
    Tensor containing zeros, with shape = size
    '''
    data = Variable(torch.zeros(size, 1))
    return data

def train_discriminator(optimizer, real_data, fake_data):
    N = real_data.size(0)
    # Reset gradients
    optimizer.zero_grad()

    # 1.1 Train on Real Data
    prediction_real = discriminator(real_data)
    # Calculate error and backpropagate
    error_real = loss(prediction_real, ones_target(N))
    error_real.backward()

    # 1.2 Train on Fake Data
    prediction_fake = discriminator(fake_data)
    # Calculate error and backpropagate
    error_fake = loss(prediction_fake, zeros_target(N))
    error_fake.backward()

    # 1.3 Update weights with gradients
    optimizer.step()

    # Return error and predictions for real and fake inputs
    return error_real + error_fake, prediction_real, prediction_fake

def train_generator(optimizer, fake_data):
    N = fake_data.size(0)
    # Reset gradients
    optimizer.zero_grad()
    # Sample noise and generate fake data
    prediction = discriminator(fake_data)
    # Calculate error and backpropagate
    error = loss(prediction, ones_target(N))
    error.backward()
    # Update weights with gradients
    optimizer.step()
    # Return error
    return error

num_test_samples = 16
test_noise = noise(num_test_samples)

# Create logger instance
#logger = Logger(model_name='VGAN', data_name='CIFAR10')
# Total number of epochs to train
num_epochs = 200

for video_path in video_paths:
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    frames_output_directory = os.path.join(base_directory, f'frames_{timestamp}/')
    video_to_frames(video_path, frames_output_directory)
    real_data = transform_images(frames_output_directory)
    N = real_data.size(0)
    for epoch in range(num_epochs):
        # 1. Train Discriminator
        real_data = images_to_vectors(real_data)
        # Generate fake data and detach 
        # (so gradients are not calculated for generator)
        fake_data = generator(noise(N)).detach()
        # Train D
        d_error, d_pred_real, d_pred_fake = train_discriminator(d_optimizer,
                                                                real_data, fake_data)

        # 2. Train Generator
        # Generate fake data
        fake_data = generator(noise(N))
        # Train G
        g_error = train_generator(g_optimizer, fake_data)
        # Log batch error
        #logger.log(d_error, g_error, epoch, n_batch, num_batches)
        # Display Progress every few batches
        if (n_batch) % 100 == 0:
            test_images = vectors_to_images(generator(test_noise))
            test_images = test_images.data

            for i, image in enumerate(test_images):
                torchvision.utils.save_image(image, os.path.join(output_directory, f'image_{i}.png'))

