import cv2
import numpy as np
from collections import Counter
from matplotlib import pyplot as plt

def read_image(image_path):
    """ Load a saved image from the specified path """
    
    # read in an image
    image = cv2.imread(image_path, cv2.COLOR_BGR2RGB)

    # convert from BGR to RGB
    imageCC = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    return(imageCC)

def show_image(image, as_pattern = False, save_path = False):
    """ Display an image, optionally as a gridden cross stitch pattern """
    
    fig, ax = plt.subplots()
    ax.imshow(image)

    if as_pattern:
        # if desired, show pattern gridding
        ax.set_xticks(np.arange(0, image.shape[0], 10))
        ax.set_yticks(np.arange(0, image.shape[1], 10))
        ax.grid(which = 'major', linewidth = 1, color = 'black')
        ax.grid(which = 'minor', linewidth = 0.5)
        ax.minorticks_on()
    else:
        ax.axis('off')
    
    if save_path:
        plt.savefig(save_path)

    plt.show()
    
def stitched_size(image, spi = 12, verbose = False):
    """ Calculate the final stitched size of an image on spi-count aida """
    
    stitchShape = image.shape[0:2]
    inchShape = (round(stitchShape[0] / spi, 1), round(stitchShape[1] / spi, 1))
    shapes = {'stitches':stitchShape, 'inches':inchShape}
    
    if verbose:
        print(f'{stitchShape[0]} by {stitchShape[1]} stitches')
        print(f'{inchShape[0]} by {inchShape[1]} inches at {spi} stitches per inch')
    return(shapes)

def resize_image(image, target):
    """ Resize an image"""
    
    if isinstance(target, float):
        # scale in fixed proportion
        newImage = cv2.resize(image, None, fx = target, fy = target)
    elif isinstance(target, tuple):
        # meet specified size
        newImage = cv2.resize(image, target)
        
    return(newImage)

def pixlts(pix_list):
    """ Convert an RGB pixel list to a string """
    
    pixString = '.'.join([str(pix_list[0]), str(pix_list[1]), str(pix_list[2])])
    return(pixString)

def pixstl(pix_string):
    """ Convert a string to an RGB pixel list """
    
    pixList = [int(x) for x in pix_string.split('.')]
    return(pixList)

def pixelate(image):
    """ Represent image colors in list and string formats"""
    
    pixels = [pixel.tolist() for row in image for pixel in row]
    pixels2 = [pixlts(pixel) for pixel in pixels]
    pix = {'pixels':pixels, 'pixelStrings':pixels2}
    return(pix)

def count_colors(image, verbose = False):
    """ Count distinct colors in an image """
    
    pix = pixelate(image)
    colorCounts = Counter(pix['pixelStrings'])
    
    if verbose:
        print(f'{len(colorCounts)} floss colors')
    return(colorCounts)

def reduce_colors(image, color_target = 10, attempts = 1):
    """ Use k-means clustering on pixel labels to identify and merge similar colors """
    
    # reshape image to one row per pixel
    pixels = image.reshape((image.shape[0] * image.shape[1], 3)).astype(np.float32)

    compactness, labels, centers = cv2.kmeans(pixels
                                              , color_target
                                              , None
                                              , (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10000, 0.0001)
                                              , attempts
                                              , cv2.KMEANS_RANDOM_CENTERS
                                             )

    # create new, still flattened image from cluster labels
    newImageFlat = np.uint8(centers)[labels.flatten()]

    # reshape image
    newImage = newImageFlat.reshape((image.shape))
    
    return(newImage)