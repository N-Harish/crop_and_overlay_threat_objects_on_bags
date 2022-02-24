# Crop and rotate threat image at 45 degree

## crop_threat_object function
* This function takes the image path as input and returns the ROI
* First read the threat images
* Then convert the image to gray scale
* Apply binary inverse thresholding to create mask
* Perform Canny Egde detection on the mask
* Perform Dilate morphological operation on the canny mask so that the boundary of threat object is clear
* Use findContour function to get the contours of the mask
* Retain the largest contour and get it's bounding box
* Crop the original image with the bounding box to get Region of Interest (ROI)

## rotate function
* Rotate function takes the image and angle as argument
* The image is ROI extracted earlier
* angle is set to 45 degree as it's mentioned in problem statement
* we get centre of the image
* then we get the rotationmatrix
* we adjust the matrix to consider translation
* Finally we use WarpAffine transform to rotate the image

## Final cropping and rotation algo
* We loop over the threat dir and apply above functions on each images
* Then pillow library is used to make background of the threat object transparent
* Then save it to cropped_threat_img directory (Note:- code will create it automatically if it doesn't exist)


# Overlaying the objects

## get_background function
* This function takes image path and returns the image and centre of largest contour
* The background image is read initially
* Then convert the image to gray scale
* Apply binary inverse thresholding to create mask
* Perform Canny Egde detection on the mask
* Perform Dilate morphological operation on the canny mask so that the boundary of threat object is clear
* Use findContour function to get the contours of the mask
* Retain the largest contour
* Using cv2.moment method we get the centre of the contour


## overlay_img function
* This function takes background image, threat image and the contour centres of background image as arguments
* Initially threat image is read using cv2.imread method with cv2.IMREAD_UNCHANGED flag to ensure the image is read in RGBA format
* Then we correct the alpha channel using cv2.split and cv2.bitwise_and methods
* Final we merge all the channels to get the image properly
* Then resize the threat image to about 30% the height of background image
* Then the centre of resized threat image is calculated which is then subtracted
  from centre of contour coordinates to get the (top, top-left) and (bottom, bottom-right) coordinate to blend
* Then overlay image is created which is just a black screen
* Using the coordinate calculated above the threat image is put over the overlay image
* Then cv2.addWeighted method is used to overlay the threat object on the background image

## Final output
* we loop over all the cropped threat images and the background images
* Then the above functions is applied iteratively
* Finally the output is saved in output directory (Note:- code will create it automatically if it doesn't exist)
