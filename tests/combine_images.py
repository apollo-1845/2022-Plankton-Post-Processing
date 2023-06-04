import cv2 as cv
import numpy as np
import math

IMAGE_SHAPE_NP = (1944, 2592, 3)
COVER_CENTRE = (2592//2, 1944//2)
COVER_RADIUS = 1944//2

CAMERA_COVER_MASK = np.full(IMAGE_SHAPE_NP, 255, dtype="uint8")
CAMERA_COVER_MASK
CAMERA_COVER_MASK = cv.circle(CAMERA_COVER_MASK, COVER_CENTRE, COVER_RADIUS, 0, -1)
# CAMERA_COVER_MASK = CAMERA_COVER_MASK != 255

# https://projects.raspberrypi.org/en/projects/astropi-iss-speed/3

def calculate_features(image_1, image_2, feature_num):
    orb = cv.ORB_create(nfeatures=feature_num)
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
    match_img = cv.drawMatches(image_1, keypoints_1, image_2, keypoints_2, matches[:100], None)
    resize = cv.resize(match_img, (1600, 600), interpolation=cv.INTER_AREA);
    cv.imshow("Matches", resize)
    cv.waitKey(0)
    cv.destroyWindow("Matches")

def get_movement_vector(image_1, image_2):
    # Get keypoints > matches
    keypoints_1, descriptors_1, keypoints_2, descriptors_2 = calculate_features(image_1, image_2, 1000)
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
        x_movements[i] = coords_2[0]-coords_1[0]
        y_movements[i] = coords_2[1]-coords_1[1]
    # Mean vector
    return np.mean(x_movements, dtype=int), np.mean(y_movements, dtype=int)
    # return np.median(x_movements).astype(int), np.median(y_movements).astype(int)

def combine_images(image_1, image_2, x_movement, y_movement):
    # TODO: Make it work with other +/- sign x_/y_movements.
    # Move image 2 so it is in the correct position, with image 1 starting at the top-left corner
    image_2 = np.hstack((image_2, np.zeros((image_2.shape[0], x_movement, image_2.shape[2]), dtype=image_2.dtype))) # Shape is y, x, depth
    image_2 = np.vstack((np.zeros((-y_movement, image_2.shape[1], image_2.shape[2]), dtype=image_2.dtype), image_2)) # Shape is y, x, depth

    # Fill image 1's mask black area with the part of image 2 in its place, then place image 1 at the top corner
    image_2[0:image_1.shape[0], x_movement:x_movement+image_1.shape[1]] = np.maximum(image_1, cv.bitwise_and(CAMERA_COVER_MASK, image_2[0:image_1.shape[0], x_movement:x_movement+image_1.shape[1]]))
    # image_2[0:image_1.shape[0], x_movement:x_movement+image_1.shape[1]] = np.maximum(image_1, image_2[0:image_1.shape[0], x_movement:x_movement+image_1.shape[1]])

    # Image 2 now refers to the combined image
    return image_2

image_1 = cv.imread("data/63.png") # 0 flags
image_2 = cv.imread("data/64.png") # 0 flags
x_movement, y_movement = get_movement_vector(image_1, image_2)
print(x_movement, y_movement)
cv.imwrite("data/temp.png", combine_images(image_1, image_2, x_movement, y_movement))