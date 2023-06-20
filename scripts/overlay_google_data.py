"""Overlay a Google Maps location annotation onto a correctly-positioned
background image. Here is how I suggest using this:

- Use the Google Static Maps API with a key set up (adults must sign the terms of
service, I think: https://developers.google.com/maps/documentation/maps-static/get-api-key)
- Use Google Maps to get the latitude or longitude of the location of an ISS photo (click a location
on the map and copy the two numbers that appear).
- Use your API key to get two images in your web browser at the following URLs (please ensure you replace {LATITUDE} and {LONGITUDE} with decimal numbers in degrees
and {YOUR API KEY} as well. You can change the zoom/center parameters to fit your photo better.
Under Layer: https://maps.googleapis.com/maps/api/staticmap?center={LATITUDE},{LONGITUDE}&zoom=7&format=png&size=432x324&maptype=roadmap&style=feature:administrative|visibility:off&style=feature:landscape|color:0x000000&style=feature:water|color:0xffffff&style=feature:road|visibility:off&style=feature:transit|visibility:off&style=feature:poi|visibility:off&key={YOUR API KEY}
Over Layer: https://maps.googleapis.com/maps/api/staticmap?center={LATITUDE},{LONGITUDE}&zoom=7&format=png&size=432x324&maptype=roadmap&style=feature:landscape|color:0x888888&style=feature:water|color:0x888888&style=feature:road|visibility:off&style=feature:transit|visibility:off&style=feature:poi|visibility:off&key={YOUR API KEY}

You may change the styling of the land contours or remove "visibility:off" parts to show roads, ferry routes (transit), etc.
as long as you keep the zoom, size and center parameters the same for each (can be changed, but has to be changed in both).

- Right-click the images loaded and save them as `{place name}_under.png` and `{place name}_over.png` in data/ignore, then edit the
under layer to add your NDVI image in the correct location (I paste the NDVI image, select "convert to sticker",
Ctrl+Z a few times, then add the image from the stickers list with 50% opacity, resize the image, then set the opacity
to 100).

- Paste your place name in the filename below, then run this script.
"""

import cv2 as cv
import numpy as np

place = "channellarge"

FILENAME_OVERLAY = f"data/ignore/{place}_over.png"
FILENAME_IN = f"data/ignore/{place}_under.png"
FILENAME_OUT = f"data/{place}_result.png"

overlay = cv.imread(FILENAME_OVERLAY)
overlay_b, overlay_g, overlay_r = cv.split(overlay) # 1 channel

image = cv.imread(FILENAME_IN)
image_b, image_g, image_r = cv.split(image) # 1 channel

image_b[(overlay_b < 130) | (overlay_b > 140)] = overlay_b[(overlay_b < 130) | (overlay_b > 140)]
image_g[(overlay_g < 130) | (overlay_g > 140)] = overlay_g[(overlay_g < 130) | (overlay_g > 140)]
image_r[(overlay_r < 130) | (overlay_r > 140)] = overlay_r[(overlay_r < 130) | (overlay_r > 140)]

image = cv.merge([image_b, image_g, image_r])

cv.imwrite(FILENAME_OUT, image)