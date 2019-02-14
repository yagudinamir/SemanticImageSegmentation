import numpy as np
from skimage.io import imread, imshow, imsave


def colorImage(input_image, output, classes_file, colors_file, saveto):

    CLASSES = open(classes_file).read().strip().split("\n")

    COLORS = open(colors_file).read().strip().split("\n")
    COLORS = [np.array(c.split(",")).astype("int") for c in COLORS]
    COLORS = np.array(COLORS, dtype="uint8")
    #print(COLORS)
    #image = imread(input_image)
    image = input_image
    #classMap = np.argmax(output, axis = 0)
    classMap = output
    mask = COLORS[classMap]
    output = ((1 * image) + (1 * mask)).astype("uint8")
    imshow(output)
    imsave(saveto, output)
