#!/usr/bin/env python3
r"""MeetUp 225 - Beginners' Python and Machine Learning - 05-Nov-2025 - Using a pretrained model

Learning objectives:
- Using pretrained models on ImageNet 1k and 21k to identify contents of a photo

Links:
- Colab:   https://colab.research.google.com/drive/12at7qFk4K5IXF5_vbtMOyQe5aTv03A9K
- Youtube: https://youtu.be/cb0V4k2Ra9Y
- Meetup:  https://www.meetup.com/beginners-python-machine-learning/events/311418695/
- Github:  https://github.com/timcu/bpaml-sessions/tree/master/online

@author D Tim Cummings

## Using a pretrained model in machine learning

- For training a model, GPU is required
- For using a pretrained model, GPU will improve performance but not required

Google colab runtime (free)

- Above go to the Runtime menu and select `Change runtime type`
- Select `T4 GPU` (and of course `Python`)

This lesson can be run using Python on your own computer or in Google Colab using nothing but a web browser.

To use Python on your own computer you will need to install Python and some third party libraries. (described here)
Alternatively install Anaconda which will install both. (not described here)

# 1. Download and install Python 3.14 from https://python.org

# 2. Create a virtual environment
mkdir bpaml225
cd bpaml225

python3 -m venv venv225         # Mac or Linux
source venv225/bin/activate     # Mac or Linux

py -m venv venv225              # Windows
venv225\Scripts\Activate.bat    # Windows

# Create a file called requirements.txt
torch
torchvision
Pillow
matplotlib
timm

# Install third party libraries listed in requirements.txt
pip install -U pip
pip install -r requirements.txt

# Run this script
python3 meetup225_tim_using_a_pretrained_model.py  # Mac or Linux
py meetup225_tim_using_a_pretrained_model.py       # Windows


## Artificial Intelligence

*the science and engineering of making intelligent machines, especially intelligent computer programs*

## Machine Learning

*branch of Artificial Intelligence which focuses on the use of data and algorithms to imitate the way humans learn, gradually improving accuracy*

## Neural Networks

*artificial neural networks mimic the human brain through a set of algorithms*

Neurons are capable of quite simple formula (linear equation) 

$output = w_1 x_1 + w_2 x_2 + w_3 x_3 + bias$

These are the weights ($w_1,w_2,w_3)$ and $bias$ of a layer that change as the model learns. 

The features $(x_1,x_2,x_3)$ are the input data to the algorithm and the $output$ is the decision made by the algorithm

## Deep Learning

*Any neural network with more than 3 layers is considered a deep learning neural network*

Output from one layer is the input to another layer. Non-linear transformations occur between layers otherwise any mix of layers could be replaced with one linear equation (layer).

GPUs are very good at thousands of parallel simple calculations and so are used extensively in deep learning. Other forms of machine learning don't need so much computing power but the input data needs to be better structured.

### References
- https://course.fast.ai
- https://pytorch.org/
"""
# Import third party libraries
# We can use standard Python libraries or 3rd party libraries.
# torchvision is part of the PyTorch library provided by Facebook,
# tensorflow is provided by Google
import torch
from torchvision import models
from torchvision import transforms
import PIL
import urllib.request
from urllib.request import urlopen
import matplotlib.pyplot as plt
import timm

# Python gives us a powerful free language for running scripts
w1 = 7
bias = 3
x1 = 2
y = w1 * x1 + bias
print('Contents of y where y = w1 * x1 + bias:', y)


# Values are remembered from one line to the next
# We can even define functions which are remembered
def my_algorithm(x):
    return w1 * x + bias


# Display results using python f strings
print(f"Results from calling function: {my_algorithm(3)=}")
print(f"Results from calling function: {my_algorithm(4)=}")

# Check if GPU is available
print(f"{torch.cuda.is_available()=}\n")
# Select the device for our computations. "cuda" has compatibility restrictions even with nvidia GPU so safer to use "cpu"
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")

# Training models can take a long time. Fortunately we can use pretrained models
# resnet50 is a 50 layer neural network trained on more than 1 million images
# we can use the model and its trained weights
# Remember that we have already imported the 'models' third party library above
pt_weights = models.ResNet50_Weights.DEFAULT
pt_model = models.resnet50(weights=pt_weights).to(device)

# Tell it we want to start evaluating images
pt_model.eval()

# Let us get an image and see how it goes.
# https://www.abc.net.au/news/2023-01-03/pomeranian-rescued-from-python-by-owner-on-sunshine-coast-beach/101823438

url_snake = "https://live-production.wcms.abc-cdn.net.au/551b5e3c441c18f273038e47032f336b"
urllib.request.urlretrieve(url_snake, "snake.jpg")
hires_snake = PIL.Image.open('snake.jpg')
hires_snake.show()

# This image is much higher resolution than the EfficientNet image set so let's
# transform the image to be similar to the ones the model was trained on.
# Previously we used the following code but this is easier now

# from torchvision import transforms
# # https://learnopencv.com/pytorch-for-beginners-image-classification-using-pre-trained-models/
# rn50_mean = [0.485, 0.456, 0.406]
# rn50_std = [0.229, 0.224, 0.225]
# transform = transforms.Compose([
#   transforms.Resize(256),
#   transforms.CenterCrop(224),
#   transforms.ToTensor(),
#   transforms.Normalize(mean=rn50_mean, std=rn50_std)])

transform = pt_weights.transforms()

# Get data in the right form for model

snake224 = transform(hires_snake)
snake_tensor = snake224.unsqueeze(0).to(device)
print(f"resnet_50 {snake224.shape=}", ' = lists of reds per pixel, greens per pixel, blues per pixel')
print(f"resnet_50 {snake_tensor.shape=}", ' = 1 image, 3 colours, 224 pixels x 224 pixels')

# Let's have a look at the image as the model will "see" it
image_np = snake224.permute(1, 2, 0).numpy()
plt.imshow(image_np)
plt.title("Raw preprocessed tensor as resnet_50 model sees it")
plt.axis('off')
plt.show()

# Calculate the probability of each class
with torch.no_grad():
    out = pt_model(snake_tensor)
print(f"{out.shape=}", '1 image identified, 1000 classes with their probability')
score_for_first_10 = str(out[0, :10]).replace('\n', '')
print(f"{score_for_first_10=}")

# Find the categories which relate to the class numbers
categories = pt_weights.DEFAULT.meta["categories"]

# out has shape (batch_size, number of classes)
# dim=1 means apply softmax across the classes dimension
probabilities = torch.softmax(out, dim=1)
# probabilities will still have a batch dimension

# Get top 5 predictions from batch 0 (probabilities[0])
top5_prob, top5_idx = torch.topk(probabilities[0], 5)

print("Top 5 guesses using ImageNet1k and resnet_50")
print(f" IDX     Prob  Class")
for i in range(5):
    print(f"{top5_idx[i]:^5}: {top5_prob[i].item()*100:6.2f}% {categories[top5_idx[i]]}")
print("\n")

# Training models can take a long time. Fortunately we can use pretrained models
# resnet50 was released in 2015. efficientnet_v2_l was released in 2021
pt_weights = models.EfficientNet_V2_L_Weights.DEFAULT
pt_model = models.efficientnet_v2_l(weights=pt_weights).to(device)

# Tell it we want to start evaluating images
pt_model.eval()
transform = pt_weights.transforms()

# Get data in the right form for model

snake224 = transform(hires_snake)
snake_tensor = snake224.unsqueeze(0).to(device)
print(f"efficientnet_v2_l {snake224.shape=}", ' = lists of reds per pixel, greens per pixel, blues per pixel')
print(f"efficientnet_v2_l {snake_tensor.shape=}", ' = 1 image, 3 colours, 480 pixels x 480 pixels')

# Let's have a look at the image as the model will "see" it
image_np = snake224.permute(1, 2, 0).numpy()
plt.imshow(image_np)
plt.title("Raw preprocessed tensor as efficientnet_v2_l model sees it")
plt.axis('off')
plt.show()

# Calculate the probability of each class
with torch.no_grad():
    out = pt_model(snake_tensor)
print(f"{out.shape=}", '1 image identified, 1000 classes with their probability')
score_for_first_10 = str(out[0, :10]).replace('\n', '')
print(f"{score_for_first_10=}")

# Find the categories which relate to the class numbers
categories = models.EfficientNet_V2_L_Weights.DEFAULT.meta["categories"]

# out has shape (batch_size, number of classes)
# dim=1 means apply softmax across the classes dimension
probabilities = torch.softmax(out, dim=1)
# probabilities will still have a batch dimension

# Get top 5 predictions from batch 0 (probabilities[0])
top5_prob, top5_idx = torch.topk(probabilities[0], 5)

print("Top 5 guesses using ImageNet1k and efficientnet_v2_l")
print(f" IDX     Prob  Class")
for i in range(5):
    print(f"{top5_idx[i]:^5}: {top5_prob[i].item()*100:6.2f}% {categories[top5_idx[i]]}")
print("\n")

# Try the same using a pretrained model from timm
# The Torch IMage Model library has many other image libraries including those based on 21k images not just 1k images
pt_model = timm.create_model('vit_base_patch16_224.augreg_in21k', pretrained=True).to(device)
pt_model.eval();
# timm uses a slightly different way to pytorch to get the necessary transforms
config = timm.data.resolve_data_config({}, model=pt_model)
transform = timm.data.create_transform(**config)
snake224 = transform(hires_snake)
snake_tensor = snake224.unsqueeze(0).to(device)
print(f"vit_base_patch16_224.augreg_in21k {snake224.shape=}", ' = lists of reds per pixel, greens per pixel, blues per pixel')
print(f"vit_base_patch16_224.augreg_in21k {snake_tensor.shape=}", ' = 1 image, 3 colours, 224 pixels x 224 pixels')
with torch.no_grad():  # always turn off gradients when doing inference. Only needed for training
    out = pt_model(snake_tensor)
probabilities = torch.softmax(out, dim=1)
# probabilities will still have a batch dimension
print(f"{probabilities.shape=}")
# Get top 5 predictions from batch 0 (probabilities[0])
top5_prob, top5_idx = torch.topk(probabilities[0], 5)
for i, p in enumerate(top5_prob):
    print(f"{i:5}: {top5_idx[i]}: {top5_prob[i].item()*100:.2f}%")

# Class names for ImageNet 21k https://github.com/google-research/big_transfer/issues/7
url = "https://storage.googleapis.com/bit_models/imagenet21k_wordnet_lemmas.txt"
import requests
response = requests.get(url)
categories = response.text.splitlines()
print(f"{len(categories)=}")
print(f"{categories[:5]=}")
print("Top 5 guesses using ImageNet21k and vit_base_patch16_224.augreg_in21k")
print(f"Num   IDX     Prob  Class")
for i, p in enumerate(top5_prob):
    print(f"{i:3}: {top5_idx[i]}: {top5_prob[i].item()*100:7.2f}% {categories[top5_idx[i]]}")
