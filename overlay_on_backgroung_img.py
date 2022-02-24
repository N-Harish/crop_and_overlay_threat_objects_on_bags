"""
@author Harish Natarajan
@project BaggageAI_CV_Hiring_Assignment
"""
import cv2
import numpy as np
from typing import Tuple
import imutils
from tqdm import tqdm
import os
import argparse


# use argparse to get the input image directory and output directory
ap = argparse.ArgumentParser()
ap.add_argument("-bg", "--background_dir", default='./background_images/', help='Enter background image path')
ap.add_argument("-th", "--threat_dir", default="./cropped_threat_img/", help="Enter cropped threat images path")
ap.add_argument("-o", "--output_dir", default='./output/', help='Enter path to save output')
arg = vars(ap.parse_args())


# get_background function takes image path as input and
# returns tuple containing image and centre of contour
def get_background(img_path: str) -> Tuple[np.array, int, int]:
    # Read the image
    image_bg = cv2.imread(img_path)
    image = image_bg.copy()
    # image = cv2.cvtColor(image, cv2.COLOR_BGR2BGRA)

    (h, w) = image.shape[:2]
    image = np.dstack([image, np.ones((h, w), dtype="uint8") * 255])

    # convert the image to gray scale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Find Inverse Binary Threshold
    _, thresh = cv2.threshold(gray, np.mean(gray), 255, cv2.THRESH_BINARY_INV)

    # perform canny edge detection
    canny = cv2.Canny(thresh, 0, 255)

    # Dilate the image to get foreground object
    edge = cv2.dilate(canny, None)

    # detect contours
    contours, hierarchy = cv2.findContours(edge, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # get largest area
    areas = [cv2.contourArea(c) for c in contours]

    # get index of largest contour
    max_index = np.argmax(areas)

    # get max area contour in cnt variable
    cnt = contours[max_index]

    # find centroid of contour
    M = cv2.moments(cnt)
    cx = int(M['m10'] / M['m00'])
    cy = int(M['m01'] / M['m00'])

    # img = cv2.drawContours(image, [cnt], -1, (0, 255, 0), 3)
    # cv2.circle(img, (cx, cy), 7, (255, 255, 0), -1)

    # return image and centroid of the largest contour
    return image, cx, cy


def overlay_img(img, threat_img, cx, cy, alpha, beta=0):
    # get height of image
    ht = img.shape[0]

    # read cropped threat image
    img1 = cv2.imread(threat_img, cv2.IMREAD_UNCHANGED)

    # below code to correct alpha channel since opencv doesn't render it properly
    (B, G, R, A) = cv2.split(img1)
    B = cv2.bitwise_and(B, B, mask=A)
    G = cv2.bitwise_and(G, G, mask=A)
    R = cv2.bitwise_and(R, R, mask=A)
    img1 = cv2.merge([B, G, R, A])

    # resize threat image to 0.25*height of background image
    img1_rs = imutils.resize(img1, height=int(0.30 * ht))

    # get centre of resized threat img
    cx1, cy1 = (img1_rs.shape[0] // 2, img1_rs.shape[1] // 2)

    # get top and left coord
    top = cx - cx1
    left = cy - cy1
    bottom_y = int(top + img1_rs.shape[0])
    right_x = int(left + img1_rs.shape[1])

    # get height and width of background image
    (h, w) = img.shape[:2]

    # create overlay image of same size to add watermark image to it
    overlay = np.zeros((h, w, 4), dtype="uint8")
    overlay[left:right_x, top:bottom_y] = img1_rs

    output_img = img.copy()

    alpha = alpha
    if beta != 0:
        beta = beta
    else:
        beta = 1 - alpha

    # perform overlay
    cv2.addWeighted(overlay, alpha, output_img, beta, 0, output_img)

    # return img
    return output_img


directory = arg['background_dir']
output_path = arg['output_dir']
threat_dir = arg['threat_dir']

ls = list(os.listdir(directory))
ls1 = list(os.listdir(threat_dir))

# generate output directory if not exists
if os.path.exists(output_path):
    print("path exists")
else:
    os.mkdir(output_path)


# loop over threat_images directory for cropping and rotating the image
for files in tqdm(ls, desc=f'Read background images '):
    # get background image and contour centroids
    img_bg, c_x, c_y = get_background(os.path.join(directory, files))

    # loop over list of images in cropped threat dir
    for cn, f in tqdm(enumerate(ls1), desc="overlay threat images"):
        img_bg1 = img_bg.copy()
        op_img_path = os.path.join(threat_dir, f)

        # call the overlay function to put threat images inside background images
        img_ov = overlay_img(img_bg1, op_img_path, c_x, c_y, alpha=0.6, beta=0.5)

        # get filename to save
        file_name = str(files).split('.')[0] + f'_{cn}' + '.jpg'
        out_pth = os.path.join(output_path, file_name)

        # save the output image
        cv2.imwrite(out_pth, img_ov)
