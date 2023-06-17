"""Generate the NDVI data as a colour map for an image, or part of it.
NDVI will appear in a range of colours from blue (lowest plant health)
to green (middle plant health) to red (high plant health), and if turned on below,
sea will be masked out black and clouds will be grey."""

import cv2 as cv
import numpy as np

USE_WHOLE_IMAGE = True # To crop the image in the process, set this to False.
MIN_X = 0 # X-coordinate of left side of crop you want (look in bottom status bar of MS Paint)
MIN_Y = 0 # Y-coordinate of top side of crop you want
MAX_X = 100 # X-coordinate of right side of crop you want
MAX_Y = 100 # Y-coordinate of bottom side of crop you want

# FILENAME_IN = "data/Europe-Greece-Turkey-Cyprus.png"
# FILENAME_IN = "data/Ireland-England-France.png"
FILENAME_IN = "data/Sudan-Ethiopia-Somalia.png" # Input image file
FILENAME_OUT = "data/Sudan-Ethiopia-Somalia_ndvi.png" # Image file to (create and / over)write to


MASK_OUT_CLOUDS = True
MASK_OUT_SEA = True

def contrast_stretch(im):
    in_min = np.percentile(im, 0)
    in_max = np.percentile(im, 100)

    out_min = 0.0
    out_max = 255.0

    out = im - in_min
    print(in_min, in_max)
    out *= ((out_min - out_max) / (in_min - in_max))
    out += in_min

    return out


image = cv.imread(FILENAME_IN)

if(not USE_WHOLE_IMAGE): # Crop
    image = image[MIN_Y:MAX_Y, MIN_X:MAX_X,0:3]

b, g, r = cv.split(image)

bottom = (r.astype(float) + b.astype(float))
bottom[bottom == 0] = 0.01
ndvi = (b.astype(float) - r) / bottom

r_float = r.astype(float)
b_float = b.astype(float)
# g_float = g.astype(float)
if(MASK_OUT_CLOUDS):
    cloud_mask = (r_float+b_float > 400) | (ndvi < 0)
if(MASK_OUT_SEA):
    sea_mask = (b < 120)

ndvi[sea_mask] = 0

ndvi = contrast_stretch(ndvi)


# Colours linked to NDVI
b[ndvi < 128] = (127 - ndvi[ndvi < 128]) * 2
g[ndvi < 128] = ndvi[ndvi < 128] * 2
r[ndvi < 128] = 0
b[ndvi >= 128] = 0
g[ndvi >= 128] = (255 - ndvi[ndvi >= 128]) * 2
r[ndvi >= 128] = (ndvi[ndvi >= 128] - 128) * 2
b[ndvi == 0] = 0
g[ndvi == 0] = 0
r[ndvi == 0] = 0

b[cloud_mask] = 128
g[cloud_mask] = 128
r[cloud_mask] = 128

b[sea_mask] = 0
g[sea_mask] = 0
r[sea_mask] = 0

image = cv.merge([b, g, r])
# image = cv.merge([ndvi,ndvi,ndvi]) - for pure greyscale NDVI image

cv.imwrite(FILENAME_OUT, image)
