from PIL import Image
import matplotlib.pyplot as plt
import torch
import torchvision.transforms as T
import numpy as np
from torchvision import models
import cv2
from optparse import OptionParser
import os
import sys

# This module uses a pytorch tutorial found here:
# https://www.learnopencv.com/pytorch-for-beginners-semantic-segmentation-using-torchvision/
# I am not claiming I came up with this code myself but I used it to perform segmentation on the images
# and remove background

# Use Fully Convolutional Network for Segmentation
fcn = models.segmentation.fcn_resnet101(pretrained=True).eval()

def decode_segmap(image, nc=21):

  label_colors = np.array([(0, 0, 0),  # 0=background
               # 1=aeroplane, 2=bicycle, 3=bird, 4=boat, 5=bottle
               (128, 0, 0), (0, 128, 0), (128, 128, 0), (0, 0, 128), (128, 0, 128),
               # 6=bus, 7=car, 8=cat, 9=chair, 10=cow
               (0, 128, 128), (128, 128, 128), (64, 0, 0), (192, 0, 0), (64, 128, 0),
               # 11=dining table, 12=dog, 13=horse, 14=motorbike, 15=person
               (192, 128, 0), (64, 0, 128), (192, 0, 128), (64, 128, 128), (192, 128, 128),
               # 16=potted plant, 17=sheep, 18=sofa, 19=train, 20=tv/monitor
               (0, 64, 0), (128, 64, 0), (0, 192, 0), (128, 192, 0), (0, 64, 128)])

  r = np.zeros_like(image).astype(np.uint8)
  g = np.zeros_like(image).astype(np.uint8)
  b = np.zeros_like(image).astype(np.uint8)

  for l in range(0, nc):
    idx = image == l
    r[idx] = label_colors[l, 0]
    g[idx] = label_colors[l, 1]
    b[idx] = label_colors[l, 2]

  rgb = np.stack([r, g, b], axis=2)
  return rgb

def segment(fcn, filepath):
  img = Image.open(filepath)

  # Convert to Tensor and Normalize
  preprocessing = T.Compose([T.ToTensor(),
                   T.Normalize(mean = [0.485, 0.456, 0.406],
                               std = [0.229, 0.224, 0.225])])
  inp = preprocessing(img).unsqueeze(0)

  # pass preprocessed image through FCN and collect output
  out = fcn(inp)['out']

  # Find classification for each image
  om = torch.argmax(out.squeeze(), dim=0).detach().cpu().numpy()

  # Create mask based on labels
  rgb = decode_segmap(om)
  return rgb

if __name__== "__main__":
    image_path = ""
    parser = OptionParser(usage="usage: %prog folder %prog out_folder", version="%prog 1.0")
    (options, args) = parser.parse_args()

    if len(args) != 2:
        parser.error("wrong number of arguments")
        sys.exit(0)

    if not os.path.isdir(args[0]):
        parser.error("Please provide folder for input images")
        sys.exit(0)
    if not os.path.isdir(args[1]):
        parser.error("Please provide folder for output images")
        sys.exit(0)

	# First segment, create RAG, and create features
    for f in os.listdir(args[0]):
        filename = os.path.join(args[0], f)
        output_filename = os.path.join(args[1], f)
        print(output_filename)
        if os.path.isfile(output_filename):
            continue
        if os.path.isfile(filename):
            if ".jpg" in filename or ".jpeg" in filename or ".png" in filename:
                img = cv2.imread(filename)
                #img = cv2.cvtColor(img.astype('uint8'),cv2.COLOR_BGR2GRAY)
                mask = segment(fcn, filename)
                mask = cv2.cvtColor(mask.astype('uint8'),cv2.COLOR_BGR2GRAY)
                img[mask == 0] = [255,192,203] # Pink
                cv2.imwrite(output_filename, img)
