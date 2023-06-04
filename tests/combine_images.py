import cv2 as cv
import numpy as np
import math

# https://projects.raspberrypi.org/en/projects/astropi-iss-speed/3

# TODO: TIF; Make more accurate; Overlay images; Check tutorial above

image_1 = cv.imread("data/63.png", 0) # 0 flags
image_2 = cv.imread("data/64.png", 0) # 0 flags

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

keypoints_1, descriptors_1, keypoints_2, descriptors_2 = calculate_features(image_1, image_2, 1000)
matches = calculate_matches(descriptors_1, descriptors_2)
display_matches(image_1, keypoints_1, image_2, keypoints_2, matches)

matches = matches[:100]

num_matches = len(matches)
x_movement = np.empty(num_matches)
y_movement = np.empty(num_matches)
for i in range(num_matches):
    match = matches[i]
    coords_1 = keypoints_1[match.queryIdx].pt
    coords_2 = keypoints_2[match.trainIdx].pt
    x_movement[i] = int(coords_2[0]-coords_1[0])
    y_movement[i] = int(coords_2[1]-coords_1[1])

print(np.mean(x_movement), np.mean(y_movement))