"""Test location"""

# Doesn't seem to work!

from PIL import Image

FILENAME = "data/0.tif"

pil_img = Image.open(FILENAME)
photo_tags = pil_img.tag_v2
print(photo_tags[270])