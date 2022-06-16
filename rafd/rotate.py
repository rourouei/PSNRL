import cv2
import dlib
import numpy as np
import os
import random
from PIL import Image

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]

IMG_JPGS = ['.jpg', '.jpeg', '.JPG', '.JPEG']
IMG_PNGS = ['.png', '.PNG']

NUMPY_EXTENSIONS = ['.npy', '.NPY']

data_dir = '/home/njuciairs/zmy/data'

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

data_dir = '/home/njuciairs/zmy/data'
rafd = data_dir + '/Rafd_aug'
rafd_dirs = []
for path in os.listdir(rafd):
    rafd_dirs.append(rafd+'/'+path)

angles = [-15, -12, -9, -6, -3, 3, 6, 9, 12, 15]

for rafd_dir in rafd_dirs:
    for angle in angles:
        dirOutput = rafd_dir + str(angle)
        print(dirOutput)
        try:
            os.makedirs(dirOutput)
        except OSError:
            pass
        for file in os.listdir(rafd_dir):
            path = os.path.join(rafd_dir, file)
            print(path)
            img = Image.open(path)
            out = img.rotate(angle)
            out = out.resize((64, 64))
            out.save(os.path.join(dirOutput, file))

