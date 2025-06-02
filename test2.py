import cv2
import numpy as np
import pytesseract
import json
import matplotlib.pyplot as plt
from ultralytics import YOLO

# === Step 1: Detect Receipt with YOLOv8 ===
model = YOLO("best.pt")  # your trained receipt detection model

img_path = "1.jpg"
image = cv2.imread(img_path)

# Run detection
results = model(image)[0]
boxes = results.boxes.xyxy.cpu().numpy()

# Assume the first box is the receipt (best confidence)
x1, y1, x2, y2 = boxes[0].astype(int)
receipt_crop = image[y1:y2, x1:x2]

# === Step 2: Preprocessing ===
gray = cv2.cvtColor(receipt_crop, cv2.COLOR_BGR2GRAY)
adaptive_thresh = cv2.adaptiveThreshold(
    gray, 255,
    cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
    cv2.THRESH_BINARY,
    15, 15
)

# === Step 3: Horizontal Segmentation ===
segments = []
start_row = 0
inside_content = False
white_row_count = 0
min_gap_height = 6

for i in range(adaptive_thresh.shape[0]):
    row = adaptive_thresh[i, :]
    if np.mean(row == 255) > 0.99:
        white_row_count += 1
    else:
        if white_row_count >= min_gap_height and inside_content:
            segment = adaptive_thresh[start_row:i - white_row_count, :]
            if segment.shape[0] > 10:
                segments.append(segment)
            inside_content = False
        if not inside_content:
            start_row = i
            inside_content = True
        white_row_count = 0

# === Step 4: Find Start and End of Product Region ===
word_list_start = ["ETTN", "ETIN", "ETİN", "Belge", "ETT", "ETTIN", "Belgo no", "ETTN NO"]
word_list_end = ["TOPLAM", "toplam"]
starting_segment, ending_segment = 0, len(segments) - 1

for i, seg in enumerate(segments):
    padded = cv2.copyMakeBorder(seg, 11, 11, 11, 11, cv2.BORDER_CONSTANT, value=255)
    text = pytesseract.image_to_string(padded, lang="eng")
    if any(word in text for word in word_list_start):
        starting_segment = i + 1
        break

for i, seg in enumerate(segments):
    padded = cv2.copyMakeBorder(seg, 11, 11, 11, 11, cv2.BORDER_CONSTANT, value=255)
    text = pytesseract.image_to_string(padded, lang="eng")
    if any(word in text for word in word_list_end):
        ending_segment = i - 1
        break

# === Step 5: Vertical Segmentation ===
combined_segment = np.vstack(segments[starting_segment:ending_segment + 1])
min_gap_width = 50
vertical_segments = []

start_col = 0
inside_col = False
white_col_count = 0

for j in range(combined_segment.shape[1]):
    col = combined_segment[:, j]
    if np.mean(col == 255) > 0.99:
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

# Final column
if inside_col and start_col < combined_segment.shape[1] - 1:
    sub_seg = combined_segment[:, start_col:]
    if sub_seg.shape[1] > 10:
        vertical_segments.append(sub_seg)

# === Step 6: OCR and Matching ===
items_list = []
custom_config = r'--psm 6'

# OCR for items
padded_items = cv2.copyMakeBorder(
    vertical_segments[0], 2, 2, 2, 2,
    cv2.BORDER_CONSTANT, value=255
)
text = pytesseract.image_to_string(padded_items, lang="eng", config=custom_config)

temp_num = ""
for line in text.split('\n'):
    line = line.strip()
    if not line or line == "\x0c":
        continue
    if any(char.isdigit() for char in line):
        temp_num = line.split(' ')[0].replace('X', '').strip()
    else:
        if not temp_num:
            items_list.append(line)
        else:
            items_list.append(f"{line} X {temp_num}")
            temp_num = ""

# OCR for prices
text2 = pytesseract.image_to_string(vertical_segments[2], lang='eng')
price_list = [line.strip() for line in text2.split('\n') if line.strip()]

# Match items and prices
item_price_dict = {}
for i in range(min(len(items_list), len(price_list))):
    item_price_dict[items_list[i]] = price_list[i]

# Save to JSON
final_data = {
    "date": "2025-05-10",
    "items": item_price_dict
}

with open("data.json", "w", encoding="utf-8") as f:
    json.dump(final_data, f, ensure_ascii=False, indent=4)

print("Final JSON Output:")
print(final_data)
