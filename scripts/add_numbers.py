from PIL import Image
from PIL import ImageDraw 
import mediapy as media
import math
import numpy as np

frames = list()

with media.VideoReader('./frame_interpolation.mp4') as reader:
    i = 57
    n = 0
    for image in reader:
        img = Image.fromarray(image)
        draw = ImageDraw.Draw(img)
        text_w, text_h = draw.textsize(str(i))
        draw.text((img.size[0]-text_w, img.size[1]-text_h), str(i),(255,255,255))
        frames.append(np.array(img))
        n+=1
        if (i == 91):
            i = 265
        elif (i == 322):
            i = 332
        elif (n % 31 == 0): # not sure why this is 31 but it works
                i+=1

media.write_video('frame_interpolation_with_numbers.mp4', frames, fps=20)
