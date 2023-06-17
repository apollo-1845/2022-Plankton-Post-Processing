import cv2 as cv
import numpy as np
import math

IMAGE_SHAPE_NP = (1944, 2592, 3)
COVER_CENTRE = (2592//2, 1944//2)
COVER_RADIUS = 1944//2

CAMERA_COVER_MASK = np.full(IMAGE_SHAPE_NP, 255, dtype="uint8")
CAMERA_COVER_MASK = cv.circle(CAMERA_COVER_MASK, COVER_CENTRE, COVER_RADIUS, 0, -1)
CAMERA_COVER_MASK_CONSERVATIVE = np.full(IMAGE_SHAPE_NP, 255, dtype="uint8")
CAMERA_COVER_MASK_CONSERVATIVE = cv.circle(CAMERA_COVER_MASK_CONSERVATIVE, COVER_CENTRE, COVER_RADIUS-100, 0, -1)
# CAMERA_COVER_MASK = CAMERA_COVER_MASK != 255

# https://projects.raspberrypi.org/en/projects/astropi-iss-speed/3

def calculate_features(image_1, image_2, feature_num):
    orb = cv.ORB_create(nfeatures=feature_num, scaleFactor=1.2, nlevels=8, edgeThreshold=50, firstLevel=0, WTA_K=2, scoreType=0, patchSize=50, fastThreshold=20)
    keypoints_1, descriptors_1 = orb.detectAndCompute(image_1, None)
    keypoints_2, descriptors_2 = orb.detectAndCompute(image_2, None)
    return keypoints_1, descriptors_1, keypoints_2, descriptors_2

def calculate_matches(descriptors_1, descriptors_2):
    # Get similar keypoints, by brute force (as suggested)
    brute_force = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)
    matches = brute_force.match(descriptors_1, descriptors_2)
    matches = sorted(matches, key=lambda x: x.distance)
    return matches

def display_matches(image_1, keypoints_1, image_2, keypoints_2, matches):
    match_img = cv.drawMatches(image_1, keypoints_1, image_2, keypoints_2, matches, None)
    resize = cv.resize(match_img, (1600, 600), interpolation=cv.INTER_AREA)
    cv.imshow("Matches", resize)
    cv.waitKey(0)
    cv.destroyWindow("Matches")

def get_movement_vector(image_1, image_2):
    # Get keypoints > matches
    keypoints_1, descriptors_1, keypoints_2, descriptors_2 = calculate_features(image_1, image_2, 10000)
    # print(len(keypoints_1), len(keypoints_2), image_2)
    matches = calculate_matches(descriptors_1, descriptors_2)
    # display_matches(image_1, keypoints_1, image_2, keypoints_2, matches)
    # Remove some with longer distance as seems to make more accurate
    # matches = matches[:100]
    # Resolve vectors from matches
    num_matches = len(matches)
    x_movements = np.empty(num_matches)
    y_movements = np.empty(num_matches)
    for i in range(num_matches):
        match = matches[i]
        coords_1 = keypoints_1[match.queryIdx].pt
        coords_2 = keypoints_2[match.trainIdx].pt
        if(CAMERA_COVER_MASK_CONSERVATIVE[int(coords_1[1])][int(coords_1[0])][0] == 255) or (CAMERA_COVER_MASK_CONSERVATIVE[int(coords_2[1])][int(coords_2[0])][0] == 255):
            x_movements[i] = -1
            y_movements[i] = -1
        else:
            x_movements[i] = coords_1[0]-coords_2[0]
            y_movements[i] = coords_1[1]-coords_2[1]
    # Mean vector
    # return np.mean(x_movements).astype(int), np.mean(y_movements).astype(int)
    return np.median(x_movements[x_movements != -1]).astype(int), np.median(y_movements[y_movements != -1]).astype(int)

def combine_images(image_1_x_offset, image_1_y_offset, image_1, image_2, x_movement, y_movement):
    # TODO: Test + fix weird sign change / inaccuracies of movement finder.

    x_offset = image_1_x_offset
    y_offset = image_1_y_offset

    # Move image 1 so it is in the correct position, with image 2 starting at the top-left corner
    if(x_movement+x_offset < 0):  # Moving beyond left side of image
        image_1 = np.hstack((np.zeros((image_1.shape[0], -x_movement-x_offset, image_1.shape[2]), dtype=image_1.dtype), image_1)) # Shape is y, x, depth
    elif(x_movement+x_offset > (image_1.shape[1]-image_2.shape[1])):  # Moving beyond right side of image
        x_offset += x_movement
        image_1 = np.hstack((image_1, np.zeros((image_1.shape[0], x_movement+x_offset-(image_1.shape[1]-image_2.shape[1]), image_1.shape[2]), dtype=image_1.dtype)))  # Shape is y, x, depth
    if (y_movement+y_offset < 0):  # Moving beyond top of image
        image_1 = np.vstack((np.zeros((-y_movement-y_offset, image_1.shape[1], image_1.shape[2]), dtype=image_1.dtype), image_1)) # Shape is y, x, depth
    elif(y_movement+y_offset > (image_1.shape[0]-image_2.shape[0])):  # Moving beyond bottom of image
        y_offset += y_movement
        image_1 = np.vstack((image_1, np.zeros((y_movement+y_offset-(image_1.shape[0]-image_2.shape[0]), image_1.shape[1], image_1.shape[2]), dtype=image_1.dtype))) # Shape is y, x, depth
    else:
        y_offset += y_movement

    # Fill image 2's mask black area with the part of image 1 in its place, then place image 2 at the top corner
    image_1[y_offset:y_offset+image_2.shape[0], x_offset:x_offset+image_2.shape[1]] = np.maximum(image_2, cv.bitwise_and(CAMERA_COVER_MASK, image_1[y_offset:y_offset+image_2.shape[0], x_offset:x_offset+image_2.shape[1]]))
    # image_1[y_offset:y_offset+image_2.shape[0], x_offset:x_offset+image_2.shape[1]] = np.maximum(image_2, image_1[y_offset:y_offset+image_2.shape[0], x_offset:x_offset+image_2.shape[1]])

    # Image 1 now refers to the combined image
    return (image_1, x_offset, y_offset)

images = list((cv.imread(f"data/{i}.png")) for i in range(74, 90))


x_movement, y_movement = 0, 0
result = images[0]
x_offset, y_offset = 0, 0

len_images = len(images)

print("/".join(str(img)[:4] for img in images))

for i in range(1, len_images):
    print(str(i) + "/" + str(len_images-1))
    x_movement, y_movement = get_movement_vector(images[i-1], images[i])
    result, x_offset, y_offset = combine_images(x_offset, y_offset, result, images[i], x_movement, y_movement)

cv.imwrite("data/Europe-Greece-Turkey-Cyprus.png", result)