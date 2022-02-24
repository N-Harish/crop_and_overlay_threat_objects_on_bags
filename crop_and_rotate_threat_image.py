"""
@author Harish Natarajan
@project BaggageAI_CV_Hiring_Assignment
"""
import os
import cv2
import numpy as np
from tqdm import tqdm
import argparse
from PIL import Image


# use argparse to get the input image directory and output directory
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image_dir", default='./threat_images/', help='Enter threat image path')
ap.add_argument("-o", "--output_dir", default="./cropped_threat_img", help="Enter path to save "
                                                                           "cropped threat images")
arg = vars(ap.parse_args())


def crop_threat_object(img_path: str) -> np.array:
    # Read the image
    image = cv2.imread(img_path)

    # keep copy of the image
    original = image.copy()

    # convert the image to gray scale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Find Inverse Binary Threshold
    _, thresh = cv2.threshold(gray, np.mean(gray), 255, cv2.THRESH_BINARY_INV)

    # perform canny edge detection
    canny = cv2.Canny(thresh, 0, 255)

    # Dilate the image to get foreground object
    edge = cv2.dilate(canny, None)

    # detect contours
    contours, _ = cv2.findContours(edge, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # get the largest area
    areas = [cv2.contourArea(c) for c in contours]

    # get index of largest contour
    max_index = np.argmax(areas)

    # get max area contour in cnt variable
    cnt = contours[max_index]

    # convert contour to rect
    x, y, w, h = cv2.boundingRect(cnt)

    # draw the rect on image
    cv2.rectangle(image, (x, y), (x + w, y + h), (36, 255, 12), 2)

    # crop the region
    roi = original[y:y + h, x:x + w]

    # return ROI
    return roi


def rotate(image, angle=45):
    # get the dimensions of the image and then determine the center
    (h, w) = image.shape[:2]
    (cX, cY) = (w // 2, h // 2)

    # calculate the rotation matrix (applying the negative of the
    # angle to rotate clockwise) and get the rotational component
    M = cv2.getRotationMatrix2D((cX, cY), -angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])

    # compute the new bounding dimensions of the image
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))

    # adjust the rotation matrix to take into account translation
    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY

    # perform the actual rotation and return the image with white background
    # , borderValue=(255, 255, 255)
    return cv2.warpAffine(image, M, (nW, nH), borderValue=(255, 255, 255))


# generate directory to save cropped threat image
if os.path.exists(arg['output_dir']):
    print("path exists")
else:
    os.mkdir(arg['output_dir'])

# get images from directory
directory = arg['image_dir']
ls = os.scandir(directory)

# degree sign
degree = u"\N{DEGREE SIGN}"

# loop over threat_images directory for cropping and rotating the image
for files in tqdm(ls, desc=f'saved threat images cropped and rotated at 45{degree} '):
    # get image path
    img = os.path.join(directory, files.name)

    # get ROI
    ROI = crop_threat_object(img)
    rot = rotate(ROI)

    # get path to save image as png
    file_name = str(files.name).split('.')[0] + '.png'
    out_pth = os.path.join(arg['output_dir'], file_name)

    # # convert background to transparent and save
    # rot = cv2.cvtColor(rot, cv2.COLOR_BGR2BGRA)
    # cv2.imwrite(out_pth, rot)

    # convert to transperent background and save using pillow
    rot = cv2.cvtColor(rot, cv2.COLOR_BGR2RGB)
    rot1 = Image.fromarray(rot)
    rot1 = rot1.convert("RGBA")

    datas = rot1.getdata()

    newData = []

    for item in datas:
        if item[0] == 255 and item[1] == 255 and item[2] == 255:
            newData.append((255, 255, 255, 0))
        else:
            newData.append(item)

    rot1.putdata(newData)
    rot1.save(out_pth, "PNG")
