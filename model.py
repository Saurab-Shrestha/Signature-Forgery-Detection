import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import pandas as pd
import warnings
# Ignore all warnings
warnings.filterwarnings("ignore")
from statistics import mean
from scipy.ndimage import rotate
import cv2

def preprocess_image(image_pth,delta=1, limit=20):
    
    def determine_score(arr, angle):
        data = rotate(arr, angle, reshape=False, order=0)
        histogram = np.sum(data, axis=1, dtype=float)
        score = np.sum((histogram[1:] - histogram[:-1]) ** 2, dtype=float)
        return histogram, score
    
    # Convert image to grayscale
    gray = image_pth.convert("L")

    # Convert grayscale image to numpy array
    img = np.array(gray)
    # Apply median blur
    blur = cv2.medianBlur(img,3)
    # blur = cv2.GaussianBlur(img,(3,3),0)
    # Define kernel for morphology operations
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))

    # Perform erosion
    erode = cv2.erode(blur, kernel, iterations=1)

    # Perform dilation
    dilate = cv2.dilate(erode, kernel, iterations=1)

    # Apply thresholding
    _, binary = cv2.threshold(dilate, 128, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    # Find the bounding box coordinates of the non-white pixels
    coords = cv2.findNonZero(binary)
    x, y, w, h = cv2.boundingRect(coords)

    # Add extra white space to the bounding box coordinates
    padding = 20  # Adjust the padding size as needed
    x -= padding
    y -= padding
    w += 2 * padding
    h += 2 * padding

    # Make sure the coordinates are within the image boundaries
    x = max(0, x)
    y = max(0, y)
    w = min(w, img.shape[1] - x)
    h = min(h, img.shape[0] - y)

    # Crop the image using the modified bounding box coordinates
    cropped_image = binary[y:y+h, x:x+w]

    # Add extra white space around the cropped image
    extra_space = np.zeros((cropped_image.shape[0] + 2 * padding, cropped_image.shape[1] + 2 * padding), dtype=np.uint8) * 255
    extra_space[padding:-padding, padding:-padding] = cropped_image
    
    scores = []
    angles = np.arange(-limit, limit + delta, delta)
    for angle in angles:
        histogram, score = determine_score(extra_space, angle)
        scores.append(score)

    best_angle = angles[scores.index(max(scores))]
    print("Best Angle: ", best_angle)
    (h, w) = extra_space.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, best_angle, 1.0)
    corrected = cv2.warpAffine(extra_space, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    # Convert the numpy array back to PIL image
    corrected = cv2.resize(corrected,(300,200))
    resized_image = Image.fromarray(corrected)

    return resized_image


class SpatialAttention(nn.Module):
    def __init__(self, in_channels):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(in_channels, 1, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Compute attention weights
        attention_scores = self.conv(x)
        attention_weights = self.sigmoid(attention_scores)

        # Apply attention to the input feature map
        attended_features = x * attention_weights

        return attended_features


class SiameseResNet(nn.Module):
    def __init__(self, model_name='resnet18', pretrained=False):
        super(SiameseResNet, self).__init__()
        self.baseModel = models.resnet18(pretrained=pretrained)

        # Experiment with different spatial sizes based on the image resolution and signature complexity
        self.attention1 = SpatialAttention(in_channels=64)  # Spatial attention for layer 1
        self.attention2 = SpatialAttention(in_channels=128)  # Spatial attention for layer 2

        self.baseModel.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.baseModel.fc = nn.Identity()

    def forward(self, x):
        out = self.baseModel.conv1(x)
        out = self.baseModel.bn1(out)
        out = self.baseModel.relu(out)
        out = self.baseModel.maxpool(out)

        out = self.attention1(self.baseModel.layer1(out))  # Applying spatial attention to layer 1
        out = self.attention2(self.baseModel.layer2(out))  # Applying spatial attention to layer 2
        out = self.baseModel.layer3(out)  # No attention for layer 3
        out = self.baseModel.layer4(out)  # No attention for layer 4

        out = self.baseModel.avgpool(out)
        out = torch.flatten(out, 1)
        return out


import torch.nn.functional as F

class LogisticSiameseRegression(nn.Module):
    def __init__(self, model):
        super(LogisticSiameseRegression, self).__init__()
        
        self.model = model
        self.fc = nn.Sequential(
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128,1)
        )
        self.sigmoid = nn.Sigmoid()
        
    def forward_once(self, x):
        out = self.model(x)
        out = F.normalize(out, p=2, dim=1)  # L2 normalization
        return out
    
    def forward(self, x1, x2):
        out1 = self.forward_once(x1)
        out2 = self.forward_once(x2)
        diff = out1 - out2
        print("Distance",torch.pairwise_distance(out1,out2))
        out = self.fc(diff)
        out = self.sigmoid(out)
        return out

