import cv2 as cv
import numpy as np

FILENAME_IN = "data/Europe-Greece-Turkey-Cyprus.png"
FILENAME_OUT = "data/Europe-Greece-Turkey-Cyprus_noclouds.png"


def contrast_stretch(im):
    in_min = np.percentile(im, 5)
    in_max = np.percentile(im, 95)

    out_min = 0.0
    out_max = 255.0

    out = im - in_min
    out *= ((out_min - out_max) / (in_min - in_max))
    out += in_min

    return out


image = cv.imread(FILENAME_IN)

b, g, r = cv.split(image)
cloud_mask = (b > 180) & (g > 140) & (r > 140)
b[cloud_mask] = 0
g[cloud_mask] = 0
r[cloud_mask] = 0

# bottom = (r.astype(float) + b.astype(float))
# bottom[bottom == 0] = 0.01
# ndvi = (b.astype(float) - r) / bottom
#
# ndvi[ndvi < 0] = 0
#
# ndvi = contrast_stretch(ndvi)
#
# ndvi[(b == 0) & (g == 0) & (r == 0)] = 0
#
# for i in range(256):
#     ndvi[0][i] = i
#
# r[ndvi < 128] = (127 - ndvi[ndvi < 128]) * 2
# g[ndvi < 128] = ndvi[ndvi < 128] * 2
# b[ndvi < 128] = 0
# r[ndvi >= 128] = 0
# g[ndvi >= 128] = (255 - ndvi[ndvi >= 128]) * 2
# b[ndvi >= 128] = (ndvi[ndvi >= 128] - 128) * 2
# r[ndvi == 0] = 0
# g[ndvi == 0] = 0
# b[ndvi == 0] = 0

image = cv.merge([b, g, r])

cv.imwrite(FILENAME_OUT, image)
