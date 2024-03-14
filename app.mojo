import torch
import torchvision.transforms as transforms
from PIL import Image

# Load the image
image_path = 'mel_spectrogram.png'
image = Image.open(image_path)

# Convert the image to grayscale
transform = transforms.Grayscale()
grayscale_image = transform(image)

# Display the grayscale image
grayscale_image.show()