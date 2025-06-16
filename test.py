import datetime
import re
from PIL import Image, ImageFilter
import numpy as np
import cv2
import pytesseract
from PIL import Image
import matplotlib.pyplot as plt
import pytesseract
from PIL import Image
import matplotlib.pyplot as plt
from rapidfuzz import fuzz, process
from collections import Counter
import pytesseract
import json


def OCR(img, pytesseract=None):
    # pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
    # Load and preprocess the image
    # img_path = "10.jpg"
    # img = cv2.imread(img_path)
    # print(img)
    # cv2.imshow('', img)
    # cv2.waitKey(0)

    # Load and preprocess the image
    # img_path = "cropped_receipts/receipt_1.jpg"
    # img = cv2.imread(img_path)

    # cv2.imshow('', img)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # cv2.imshow('', gray)

    # Adaptive thresholding to binarize
    adaptive_thresh = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        15,
        15
    )

    # cv2.imshow('', adaptive_thresh)

    def process_all_white_layers(image, white_threshold=0.98):
        h, w = image.shape
        layer = 0

        while True:
            top = layer
            bottom = h - layer
            left = layer
            right = w - layer

            # Stop if window is too small
            if bottom <= top or right <= left:
                break

            # Get border pixels for current layer
            top_row = image[top, left:right]
            bottom_row = image[bottom - 1, left:right]
            left_col = image[top + 1:bottom - 1, left]
            right_col = image[top + 1:bottom - 1, right - 1]

            border_pixels = np.concatenate((top_row, bottom_row, left_col, right_col))

            # masked = np.ones_like(image, dtype=np.uint8) * 255
            # masked[top:bottom, left:right] = image[top:bottom, left:right]
            # plt.figure(figsize=(8, 8))
            # plt.imshow(masked, cmap='gray')
            # plt.title(f"Layer {layer}")
            # plt.axis("off")
            # plt.show()

            # Check if at least 99% of border pixels are white
            white_ratio = np.mean(border_pixels == 255)

            if white_ratio > white_threshold:
                print(f"Layer {layer} is {white_ratio:.2%} white. Accepting layer.")

                # Create a mask that keeps only current layer and inside
                masked = np.ones_like(image, dtype=np.uint8) * 255
                masked[top:bottom, left:right] = image[top:bottom, left:right]

                # cv2.imshow('', masked)
                # cv2.waitKey(0)

                return masked
            else:
                print(f"Layer {layer} contains non-white pixels. Stopping.")

            layer += 1

    adaptive_thresh = process_all_white_layers(adaptive_thresh)

    # Count white and black pixels
    white_count = np.sum(adaptive_thresh == 255)
    black_count = np.sum(adaptive_thresh == 0)
    print("White pixels:", white_count)
    print("Black pixels:", black_count)

    # Horizontal segmentation based on white rows
    segments = []
    start_row = 0
    inside_content = False
    white_row_count = 0
    min_gap_height = 3  # Minimum number of mostly-white rows to treat as a gap

    for i in range(adaptive_thresh.shape[0]):
        row = adaptive_thresh[i, :]
        if np.mean(row == 255) > 0.99:
            white_row_count += 1
        else:
            if white_row_count >= min_gap_height and inside_content:
                # We just finished a white gap and have content
                segment = adaptive_thresh[start_row:i - white_row_count, :]
                if segment.shape[0] > 10:
                    segments.append(segment)
                inside_content = False
            if not inside_content:
                start_row = i
                inside_content = True
            white_row_count = 0  # reset because we hit a non-white row

    # Show horizontal segments
    for idx, seg in enumerate(segments):
        plt.figure(figsize=(8, 2))
        plt.imshow(seg, cmap='gray')
        plt.title(f"Horizontal Segment {idx + 1}")
        plt.axis("off")
        plt.show()
        print()

    # for i, segment in enumerate(segments):
    #     cv2.imshow('', segment)
    #     print(i)
    # text = pytesseract.image_to_string(segments[5], lang='eng')
    # cv2.imshow('', segments[5])
    # print(text)

    word_list = ["ETTN", "ETIN", "ETİN", "ETT", "ETTIN", "ETTN NO", "EVTN", "EVTW"]

    starting_segement, ending_segment = 0, 0
    found_pos = False

    # # find the segement that has the word "ETTN" in it
    for i, vseg in enumerate(segments):
        print(i)
        # add padding
        vseg = cv2.copyMakeBorder(
            vseg,
            top=11,
            bottom=11,
            left=11,
            right=11,
            borderType=cv2.BORDER_CONSTANT,
            value=255  # white border for binary image
        )
        text = pytesseract.image_to_string(vseg, lang='eng')
        print(text)
        words = text.split()
        for word in words:
            best_match, score, _ = process.extractOne(word, word_list, scorer=fuzz.ratio)
            if score > 80:
                # cv2.imshow('', vseg)
                # cv2.waitKey(0)
                starting_segement = i + 1
                print(i)
                found_pos = True
                break
        if found_pos:
            break

    found_pos = False
    word_list2 = ["TOPKDY", "TOPKFV", "TOPLAM", "toplam", "TGPLAM"]

    for i, vseg in enumerate(segments):
        print(i)
        vseg = cv2.copyMakeBorder(
            vseg,
            top=11,
            bottom=11,
            left=11,
            right=11,
            borderType=cv2.BORDER_CONSTANT,
            value=255)
        text = pytesseract.image_to_string(vseg, lang='eng')
        print(text)
        words = text.split()
        for word in words:
            best_match, score, _ = process.extractOne(word, word_list2, scorer=fuzz.ratio)
            if score > 80:
                # cv2.imshow('', vseg)
                ending_segment = i -1
                print(i)
                found_pos = True
                break
        if found_pos:
            break

    print("Starting and Ending segment no:", starting_segement, ending_segment)

    # ============================================
    # Vertical segmentation on one horizontal segment (e.g., first one)
    # ============================================
    min_gap_width = 70  # Minimum number of white columns to trigger segmentation
    vertical_segments = []

    # Combine relevant horizontal segments into a single block
    combined_segment = np.vstack(segments[starting_segement:ending_segment])

    # Vertical segmentation logic
    start_col = 0
    inside_col = False
    white_col_count = 0

    for j in range(combined_segment.shape[1]):
        col = combined_segment[:, j]
        if np.mean(col == 255) > 0.95:  # Mostly white column
            white_col_count += 1
        else:
            if white_col_count >= min_gap_width and inside_col:
                sub_seg = combined_segment[:, start_col:j - white_col_count]
                if sub_seg.shape[1] > 10:
                    vertical_segments.append(sub_seg)
                inside_col = False
            if not inside_col:
                start_col = j
                inside_col = True
            white_col_count = 0

    # Handle final vertical segment
    if inside_col and start_col < combined_segment.shape[1] - 1:
        sub_seg = combined_segment[:, start_col:]
        if sub_seg.shape[1] > 10:
            vertical_segments.append(sub_seg)

    # Show results
    for i, vseg in enumerate(vertical_segments):
        plt.figure(figsize=(2, 4))
        plt.imshow(vseg, cmap='gray')
        plt.title(f"Vertical Sub-Segment {i + 1}")
        plt.axis("off")
        plt.show()

    border_size = 15  # pixels

    # read all the segments using pyteseract and count the number of characters
    counter = {}
    for i, vseg in enumerate(vertical_segments):
        padded_vseg = cv2.copyMakeBorder(vseg,
                                        top=border_size,
                                        bottom=border_size,
                                        left=border_size,
                                        right=border_size,
                                        borderType=cv2.BORDER_CONSTANT,
                                        value=255  # white border for binary image
                                        )
        text = pytesseract.image_to_string(padded_vseg, lang='eng')
        count = len(text)

        # add to dictionary by id and count
        counter[i] = count

        print(f"Segment {i + 1}: {text}")

    print(f"Counter: {counter}")

    # products segmetn is the entry with the highest number of characters

    temp_counter = Counter(counter)
    # Get the two highest (key, value) pairs
    top_two = temp_counter.most_common(2)

    # Extract just the keys
    top_keys = [k for k, v in top_two]

    print("Top 2 keys:", top_keys)

    products_segment = vertical_segments[top_keys[0]]
    plt.figure(figsize=(8, 2))
    plt.imshow(products_segment, cmap='gray')
    plt.title(f"Products Segment")
    plt.axis("off")
    plt.show()
    # cv2.imshow('', products_segment)
    # cv2.waitKey(0)
    prices_segment = vertical_segments[top_keys[1]]
    plt.figure(figsize=(8, 2))
    plt.imshow(prices_segment, cmap='gray')
    plt.title(f"Prices Segment")
    plt.axis("off")
    plt.show()
    # cv2.imshow('', prices_segment)
    # cv2.waitKey(0)

    padded_img = cv2.copyMakeBorder(products_segment,
                                    top=border_size,
                                    bottom=border_size,
                                    left=border_size,
                                    right=border_size,
                                    borderType=cv2.BORDER_CONSTANT,
                                    value=255)
    # cv2.imshow('', padded_img)

    # TEST________________________________________________________
    temp_segments = []
    start_row = 0
    inside_content = False
    white_row_count = 0
    min_gap_height = max(1, int(padded_img.shape[0] * 0.005))  # Use percent or fixed value

    for i in range(padded_img.shape[0]):
        row = padded_img[i, :]
        if np.mean(row == 255) > 0.99:
            white_row_count += 1
        else:
            if white_row_count >= min_gap_height and inside_content:
                segment = padded_img[start_row:i - white_row_count, :]
                if segment.shape[0] > 10:
                    temp_segments.append(segment)
                inside_content = False
            if not inside_content:
                start_row = i
                inside_content = True
            white_row_count = 0

    # Add final segment if still inside content at end of image
    if inside_content:
        segment = padded_img[start_row:, :]
        if segment.shape[0] > 10:
            temp_segments.append(segment)

    # Show horizontal segments
    for idx, seg in enumerate(temp_segments):
        plt.figure(figsize=(8, 2))
        plt.imshow(seg, cmap='gray')
        plt.title(f"Horizontal Segment {idx + 1}")
        plt.axis("off")
        plt.show()
        print()

    text_str = ""
    for segment in temp_segments:
        border_size = 15  # pixels
        padded_segment = cv2.copyMakeBorder(segment,
                                            top=border_size,
                                            bottom=border_size,
                                            left=border_size,
                                            right=border_size,
                                            borderType=cv2.BORDER_CONSTANT,
                                            value=255  # white border for binary image
                                            )

        plt.figure(figsize=(8, 2))
        plt.imshow(padded_segment, cmap='gray')
        plt.title(f"Horizontal Segment {idx + 1}")
        plt.axis("off")
        plt.show()

        # cv2.imshow('', padded_segment)
        # if line is mostly numbers, it's probably a price
        custom_config = r'--psm 6'  # 6 = Assume a uniform block of text
        text = pytesseract.image_to_string(padded_segment, lang='eng', config=custom_config)
        print(f"Segment {idx + 1}: {text}")

        # if line is mostly numbers, it's probably a price
        if len(text.split()) > 0 and sum(c.isdigit() for c in text) / len(text) < 0.5:
            text = pytesseract.image_to_string(padded_segment, lang='tur', config=custom_config)
            print(text)
            text_str += text + "\n"
        else:
            print(text)
            text_str += text + "\n"

    # TEST________________________________________________________

    # Normal Teseracrt
    # Step 1: Extract items (in Turkish)
    print(f"Text: {text_str}")
    items_list = []
    # custom_config = r'--psm 6'  # 6 = Assume a uniform block of text
    # text = pytesseract.image_to_string(padded_img, lang='tur', config=custom_config)
    # print(text)

    #TODO: 6 X 6.50 SU 1.5 LT

    temp_num = ""
    for line in text_str.split('\n'):
        line = line.strip()
        print(f"Line: {line}")
        if not line or line == "\x0c" or line == "\n":
            continue
        # if first character isdigit
        #TODO: add a seperation for both KG, L and normal ones also G if have time
        if len(line.split()) > 0 and sum(c.isdigit() for c in line) / len(line) > 0.4 or ' kg ' in line or ' ka ' in line or ' K ' in line or ' k ' in line:
            # if first character is not a digit, remove it
            if ' kg ' in line or ' ka ' in line or ' K ' in line or ' k ' in line:
                line = line.replace('kg', '').strip()
                print(f"Line with kg: {line}")
                # get everything before ' ka' or' kg '
                if ' ka ' in line :
                    line = line.replace(' ka ', ' kg ').strip()
                elif ' K ' in line:
                    line = line.replace(' K ', ' kg ').strip()
                elif ' k ' in line:
                    line = line.replace(' k ', ' kg ').strip()
                temp_num = line.split(' kg ')[0] + ' kg'
                print(f"Temp Num with kg : {temp_num}")
            else:
                if not line[0].isdigit():
                    line = line[2:]
                print(f"Line with Number: {line}")
                # line = line.replace(' ', '').strip()
                temp_num = line.split(' ')[0].replace('X', '').strip()
        else:
            if not temp_num:
                items_list.append(line)
            else:
                items_list.append(f"{line} X {temp_num}")
                temp_num = ""

    print("Items List:", items_list)

    # Extract prices (in English OCR)
    # cv2.imshow('', vertical_segments[2])
    padded_prices_segment = cv2.copyMakeBorder(prices_segment,
                                            top=border_size,
                                            bottom=border_size,
                                            left=border_size,
                                            right=border_size,
                                            borderType=cv2.BORDER_CONSTANT,
                                            value=255  # white border for binary image
                                            )
    text2 = pytesseract.image_to_string(padded_prices_segment, lang='eng')
    price_list = [line.strip() for line in text2.split('\n') if line.strip()]
    print("Price List:", price_list)
    # Match items and prices safely
    # item_price_dict = {}
#
    # # for when it doesn't read any price
    # for i in range(min(len(items_list), len(price_list))):
    #     print(f"Matching {items_list[i]} with {price_list[i]}")
    #     if price_list[i] is None:
    #         item_price_dict[items_list[i]] = 0
    #     else:
    #         item_price_dict[items_list[i]] = price_list[i]
    # print("Item-Price Dictionary:", item_price_dict)

    # Add date (dummy or dynamic)

    def parse_items(items_list, price_list=None, userid=13):
        print("Parsing items...")
        item_entries = []

        for i in range(len(items_list)):
            item_str = items_list[i]
            price = price_list[i] if price_list and i < len(price_list) and price_list[i] else "0.0"

            print(f"Processing item {item_str} with price {price}")

            # Regex to match numbers with dot/comma and optional unit
            match = re.search(r'(?i)(\d+(?:[.,]\d+)?)(?:\s*)([a-zA-ZçÇğĞıİöÖşŞüÜ]+)?$', item_str.strip())

            if match:
                amount = match.group(1).replace(",", ".")
                unit = match.group(2).upper() if match.group(2) else "ADET"
            else:
                amount = "1"
                unit = "ADET"

            name_part = item_str[:match.start()] if match else item_str
            item_name = name_part.strip().upper()

            date_str = datetime.datetime.now().strftime("%d/%m/%Y")
            item_entries.append([item_name, amount, unit, price, date_str])

        return {
            "userid": userid,
            "items": item_entries
        }

    result = parse_items(items_list, price_list)
    print(result)

    json_data = json.dumps(result, ensure_ascii=False, indent=4)
    return json_data