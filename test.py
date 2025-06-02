from PIL import Image, ImageFilter
import numpy as np
import cv2
import pytesseract
from PIL import Image

if __name__ == '__main__':


    img = cv2.imread('temp.jpg', cv2.IMREAD_GRAYSCALE)

    adaptive_thresh = cv2.adaptiveThreshold(
        img,  # input image (must be grayscale)
        255,  # maximum value assigned to pixels
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,  # algorithm to calculate threshold
        cv2.THRESH_BINARY,  # threshold type
        15,  # block size (small odd number like 11 or 15)
        15  # constant subtracted from mean
    )

    # Save result or use cleaned image for OCR
    cv2.imwrite('adaptive_thresholded.jpg', adaptive_thresh)


    # convert nparray to PIL image
    test_image = Image.fromarray(adaptive_thresh)

    test_image = test_image.convert("L")
    image = test_image.filter(ImageFilter.FIND_EDGES)
    image.show()

    # find the horizontal lines
    horizontal_lines = image.filter(ImageFilter.MinFilter(size=5))
    horizontal_lines.show()

    # Set Turkish language
    # text = pytesseract.image_to_string(adaptive_thresh, lang='tur')
#
    # print("Detected Text:")
    # print(text)