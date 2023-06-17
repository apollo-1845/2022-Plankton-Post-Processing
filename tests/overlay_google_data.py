import cv2 as cv
import numpy as np

FILENAME_OVERLAY = "data/test_overlay.png"
FILENAME_IN = "data/62.png"
FILENAME_OUT = "data/test_result.png"

overlay = cv.imread(FILENAME_OVERLAY)
overlay_b, overlay_g, overlay_r = cv.split(overlay) # 1 channel

image = cv.imread(FILENAME_IN)
image_b, image_g, image_r = cv.split(image) # 1 channel

image_b[(overlay_b < 130) | (overlay_b > 140)] = overlay_b[(overlay_b < 130) | (overlay_b > 140)]
image_g[(overlay_g < 130) | (overlay_g > 140)] = overlay_g[(overlay_g < 130) | (overlay_g > 140)]
image_r[(overlay_r < 130) | (overlay_r > 140)] = overlay_r[(overlay_r < 130) | (overlay_r > 140)]

image = cv.merge([image_b, image_g, image_r])

cv.imwrite(FILENAME_OUT, image)