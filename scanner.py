import cv2
import sys, getopt
import numpy as np
import matplotlib.pyplot as plt
import os
from skimage.filters import threshold_local


def findContours(org_image):
    """
    Find all Contours in image
    :param image: Gray Scale Image
    :return: Image Contours
    """
    thres, imgB = cv2.threshold(org_image, 130, 255, cv2.THRESH_BINARY)
    cnts, _ = cv2.findContours(imgB, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    return cnts


def getBiggestContur(image):
    """
    Finds all conturs in given image and returns the biggest one with 4 points
    :param image: Grayscale image
    :return: Biggest Contour points
    """
    cnts = findContours(image)
    contours = sorted(cnts, key=cv2.contourArea, reverse=True)

    for cnt in contours:
        p = cv2.arcLength(cnt, True)
        cnt_approx = cv2.approxPolyDP(cnt, 0.1 * p, closed=True)
        if len(cnt_approx) == 4:
            return cnt_approx


def edge_detection(image, t1=45, t2=60):
    """
    Canny Edge Detection on given image
    :param image: image to run Canny on
    :param t1: Threshold low
    :param t2: Threshold high
    :return: Edge Detected Image
    """
    return cv2.Canny(image, threshold1=t1, threshold2=t2, apertureSize=3, L2gradient=False)


def warpPrespective(image, bContour, pts):
    """
    Create New Image From Contour
    :param bContour: points of original rectangle in image
    :param pts: new position of original points
    :return: new image placing biggest contour from original image as the image
    """
    rows, cols = image.shape[:2]
    M = cv2.getPerspectiveTransform(bContour, pts)
    return cv2.warpPerspective(image, M, (rows, cols))


def showimage(image, mode=cv2.COLOR_BGR2RGB):
    """
    Plots The image with given mode
    :param image: image to be shown
    :param mode: flag for CvtColor
    :return: None
    """
    plt.imshow(cv2.cvtColor(image, mode))
    plt.show()
    cv2.waitKey(0)  # Avoid Python kernel from crashing


def order_points(pts):
    """
    Order Points To Represent a Rectangle
    :param pts: 4 points
    :return: points with new order
    """
    pts = pts.reshape((4, 2))
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    return rect


def scan_file(source, dest):
    """
    Scans a single file from source and saves to dest
    :param source: Path to image
    :param dest: Destination of output image
    :return: None
    """

    # Load image to be scanned
    image = cv2.imread(source, cv2.IMREAD_COLOR)
    save_orginal_dim = image.shape

    # PreProcess
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Save original image
    org_image = image.copy()

    image = cv2.GaussianBlur(image, (11, 11), 0)
    image = edge_detection(image)
    image = cv2.dilate(image, np.ones((5, 5), 'uint8'), iterations=2)

    # Get 4 Points Of Object
    bContour = getBiggestContur(image)
    # Draw Contours to image
    # drawcnt = cv2.drawContours(org_image, [bContour], -1, (0, 255, 0), 20)
    # cv2.imwrite('drawContours.jpg', drawcnt)

    # Create New Image From Contour
    height = org_image.shape[1]
    width = org_image.shape[0]
    bContour = order_points(bContour)
    pts1 = np.float32([[0, 0], [0, height], [width, height], [width, 0]])
    pts1 = order_points(pts1)
    shiftedImage = warpPrespective(org_image, bContour, pts1)

    # Black and white image
    thresh = threshold_local(shiftedImage, 35, offset=10)
    binaryImage = (shiftedImage > thresh).astype("uint8") * 255

    dim = (save_orginal_dim[1], save_orginal_dim[0])
    binaryImage = cv2.resize(binaryImage, dim, interpolation=cv2.INTER_AREA)
    # Save output to destination
    cv2.imwrite(dest, binaryImage)


def scan_complete_folder(source):
    """
    loads all images ending with '*.jpg' from source
    and runs the scanner on them
    :param source: Path to folder
    :return: None
    """
    output_dir = os.path.join(source, "Output")
    os.makedirs(output_dir, exist_ok=True)
    for file in os.listdir(source):
        if file[-3:] == "jpg":
            image_to_open = '' + source + file
            print('Scanning ' + image_to_open)
            scan_file(image_to_open, str(output_dir + '/' + file))


def help():
    print("'python scanner.py <OPTIONS> <SOURCE> <DESTINATION>'")
    print("if DESTINATION is not specified program will save as output.jpg")
    print("     -f: Execute scanner on all *.jpg files in given folder ")


if __name__ == '__main__':
    try:
        options_short = "f:h:"
        options_long = ["help", "folder"]
        options, args = getopt.getopt(sys.argv[1:], options_short, options_long)
        options_flags = options.copy()

        for o, a in options:
            if o == "--help":
                help()
                sys.exit()
            elif o in ("-f", "--folder"):
                scan_complete_folder(a)

        if len(args) > 0:
            source = args[0]
            dest = args[1] if len(args) == 2 else "output.jpg"
            # Run Main Code
            scan_file(source, dest)

        if not len(options) and not len(args):
            source = sys.argv[1]
            dest = sys.argv[2] if len(sys.argv) == 3 else "output.jpg"
            # Run Main Code
            scan_file(source, dest)
        exit(0)

    except Exception as e:
        print(e)
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(exc_type, fname, exc_tb.tb_lineno)
        exit(1)
