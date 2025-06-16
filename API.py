from flask import Flask, request, jsonify
import os
from main import Start_yolo
import pytesseract
import requests


pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
pytes = pytesseract
app = Flask(__name__)

# Directory to temporarily save uploaded images
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


def replace_item_names(original_data, temiz_urunler):
    items = original_data["items"]
    if len(items) != len(temiz_urunler):
        raise ValueError("Mismatch between original items and cleaned names count.")

    # Replace each item name
    for i in range(len(items)):
        items[i][0] = temiz_urunler[i]

    return {
        "userid": original_data["userid"],
        "items": items
    }


@app.route("/upload", methods=["POST"])
def upload_image():
    if "image" not in request.files:
        return jsonify({"error": "No image part in the request"}), 400

    image = request.files["image"]
    if image.filename == "":
        return jsonify({"error": "No selected image"}), 400

    image_path = 'cropped_receipts/receipt.jpg'
    image.save(image_path)

    try:
        # Call your YOLO detection
        res = Start_yolo(image_path, pytes)

        # send to the ip address http://192.168.1.100:5001/ocr-düzenle
        url = "http://192.168.1.100:5001/ocr-düzenle"
        response = requests.post(url, json=res)
        print(f"Result: {res}")

        print(f"Response from OCR and LLM servers: {response.text}")
        response_json = response.json()
        save_ocr(response_json)
        return jsonify({"message": "Image received and processed"}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/single_upload", methods=["POST"])
def single_upload():
    print("Single upload")
    print(f"Request: {request.json}")
    """
    This function is used to upload a single item in the following format:
        {
        "items": [
        [
          "su",
          "2",
          "L",
          "0.0",
          "23/06/2025"
        ],
        "userid": 13
        }
    :return:
    """
    if "items" not in request.json:
        return jsonify({"error": "No items part in the request"}), 400

    items = request.json["items"]
    userid = request.json["userid"]

    try:
        # change the recived data to match the needed format
        res = {
            "items": items,
            "userid": userid
        }
        # send to the ip address http://192.168.1.100:5001/ocr-düzenle
        url = "http://192.168.1.100:5001/ocr-düzenle"
        response = requests.post(url, json=res)
        print(f"Result: {res}")

        print(f"Response from OCR and LLM servers: {response.text}")
        response_json = response.json()
        save_ocr(response_json)
        return jsonify({"message": "Image received and processed"}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# @app.route('/items', methods=['POST'])
# def receive_items():
#     data = request.get_json()
#
#     print("Received items:", data)
#
#     for item in data:
#         print(item)
#
#     # Validate input
#     if not data or 'items_list' not in data:
#         return jsonify({'error': 'Missing items_list in request'}), 400
#
#     # items_list = data['items_list']
# #
#     # # Log or print for now
#     # print("Received items:", items_list)
#
#     # Just echo back for now
#     return jsonify({'message': 'Items received successfully', 'items_received': items_list}), 200


@app.route("/ping", methods=["GET"])
def ping():
    print("pong")
    return "pong"


def save_ocr(json_file):
    import requests
    import json

    print("Received items:", json_file)
    print(json.dumps(json_file, ensure_ascii=False, indent=2))

    url = "http://192.168.1.100:8081/api/ocr/convert"

    # Only parse if it's a string
    if isinstance(json_file, str):
        json_file = json.loads(json_file)

    response = requests.post(url, json=json_file)
    print(response.text)
    return response.text


if __name__ == "__main__":
    # test pytesseract
    # print(pytesseract.image_to_string(r'cropped_receipts\receipt_1.jpg'))
    # img = "cropped_receipts/receipt_1.jpg"
    # Start_yolo(img, pytes)
    app.run(host="0.0.0.0", port=5002)
