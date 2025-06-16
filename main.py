from ultralytics import YOLO
import cv2
import torch
import numpy as np
import os
import test


def image_preprocessing(image_path, save_path):
    """
    Preprocesses the input image before running it through the YOLOv8 model.
    """
    x = 0
    for img in os.listdir(image_path):
        if x < 10:
            img_path = os.path.join(image_path, img)
            img = cv2.imread(img_path)
            img = cv2.resize(img, (640, 640))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = img / 255.0

            # add random blur
            img = cv2.GaussianBlur(img, (5, 5), 0)

            # add random noise
            noise = np.random.normal(0, 0.02, img.shape)
            img = img + noise
            img = np.clip(img, 0, 1)

            # add random brightness
            img = img * np.random.uniform(0.5, 1.5)
            img = np.clip(img, 0, 1)

            # add random contrast
            mean = np.mean(img)
            img = (img - mean) * np.random.uniform(0.5, 1.5) + mean
            img = np.clip(img, 0, 1)

            # add random rotation
            angle = np.random.uniform(-10, 10)
            M = cv2.getRotationMatrix2D((img.shape[1] / 2, img.shape[0] / 2), angle, 1)
            img = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]))

            # add random translation
            x = np.random.uniform(-0.1, 0.1) * img.shape[1]
            y = np.random.uniform(-0.1, 0.1) * img.shape[0]
            M = np.float32([[1, 0, x], [0, 1, y]])
            img = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]))

            # add random scaling
            scale = np.random.uniform(0.9, 1.1)
            img = cv2.resize(img, (0, 0), fx=scale, fy=scale)
            img = cv2.resize(img, (640, 640))

            # add random flip
            if np.random.random() < 0.5:
                img = cv2.flip(img, 1)

            img = torch.tensor(img).permute(2, 0, 1).float()
            img = img.unsqueeze(0)

            # file_name =  image_path + img + '.pt'
            # torch.save(img, os.path.join(save_path, img))
            x += 1

            img = img.squeeze(0).permute(1, 2, 0).numpy()
            cv2.imshow('image', img)
            cv2.waitKey(0)


def train_yolo():
    """
    Trains the YOLOv8 model using a dataset in YOLO format.
    Ensure that your dataset is in the format expected by YOLOv8.
    """
    model = YOLO("yolov8n.yaml")  # Load YOLO model architecture
    model.train(
        data="data.yaml",  # Path to dataset YAML file
        epochs=50,  # Adjust based on dataset size
        imgsz=640,
        batch=16,
        project="yolov8_training",
        name="receipt_detector"
    )


def load_yolo_model(weights_path):
    """Load the trained YOLOv8 model from the weights file."""
    model = YOLO(weights_path)
    return model


def detect_receipts(model, source="http://<phone-ip>:8080/video", conf=0.5, frame_width=640, frame_height=480):
    cap = cv2.VideoCapture(source)

    if not cap.isOpened():
        print("Error: Could not open video source.")
        return

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture frame.")
            break

        # Resize the frame to avoid zoom-in issues
        frame = cv2.resize(frame, (frame_width, frame_height))

        # Run YOLOv8 model on the resized frame
        results = model(frame)

        # Visualize results
        for r in results:
            for box in r.boxes.data:
                x1, y1, x2, y2, conf, cls = map(int, box[:6])
                label = f"{model.names[cls]} {conf:.2f}"
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        cv2.imshow("YOLOv8 Receipt Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):  # Press 'q' to quit
            break

    cap.release()
    cv2.destroyAllWindows()


def detect_and_save_receipts_from_image(model, image_path, save_dir="cropped_receipts", conf=0.2):
    # Load the image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not read image from {image_path}")
        return None

    # Create save directory
    os.makedirs(save_dir, exist_ok=True)
    results = model(image, conf=conf)

    img_counter = 0
    for r in results:
        for box in r.boxes.data:
            x1, y1, x2, y2, confidence, cls = box[:6]
            if confidence >= conf:
                x1, y1, x2, y2 = map(int, (x1, y1, x2, y2))
                cropped = image[y1:y2, x1:x2]

                if cropped.size > 0:
                    save_path = os.path.join(save_dir, f"receipt_{img_counter}.jpg")
                    cv2.imwrite(save_path, cropped)
                    print(f"[INFO] Saved {save_path}")
                    img_counter += 1
                    return cropped

    if img_counter == 0:
        print("[INFO] No receipts detected.")
        return None
    return None


def detect_and_save_receipts_with_crops(model, image_path, save_dir="cropped_receipts", conf=0.2, crop_size=(512, 512), step_size=256):
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not read image from {image_path}")
        return None

    os.makedirs(save_dir, exist_ok=True)
    img_height, img_width = image.shape[:2]
    img_counter = 0

    # Try detecting from full image first
    results = model(image, conf=conf)
    for r in results:
        for box in r.boxes.data:
            x1, y1, x2, y2, confidence, cls = box[:6]
            if confidence >= conf:
                x1, y1, x2, y2 = map(int, (x1, y1, x2, y2))
                cropped = image[y1:y2, x1:x2]
                if cropped.size > 0:
                    save_path = os.path.join(save_dir, f"receipt_full_{img_counter}.jpg")
                    cv2.imwrite(save_path, cropped)
                    print(f"[INFO] Detected in full image. Saved {save_path}")
                    return cropped

    # If no detection in full image, try cropped windows
    print("[INFO] No detection in full image. Trying sliding window...")
    for y in range(0, img_height - crop_size[1] + 1, step_size):
        for x in range(0, img_width - crop_size[0] + 1, step_size):
            crop = image[y:y+crop_size[1], x:x+crop_size[0]]
            results = model(crop, conf=conf)
            for r in results:
                for box in r.boxes.data:
                    x1, y1, x2, y2, confidence, cls = box[:6]
                    if confidence >= conf:
                        x1, y1, x2, y2 = map(int, (x1, y1, x2, y2))
                        detected_crop = crop[y1:y2, x1:x2]
                        if detected_crop.size > 0:
                            save_path = os.path.join(save_dir, f"receipt_crop_{img_counter}.jpg")
                            cv2.imwrite(save_path, detected_crop)
                            print(f"[INFO] Detected in crop. Saved {save_path}")
                            return detected_crop

    print("[INFO] No receipts detected in sliding window crops.")
    return None


def shift_and_zoom_detect(model, image_path, save_dir="cropped_receipts", conf=0.2):
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not read image from {image_path}")
        return None

    os.makedirs(save_dir, exist_ok=True)
    img_h, img_w = image.shape[:2]
    img_counter = 0

    def try_detect(img, label):
        results = model(img, conf=conf)
        for r in results:
            for box in r.boxes.data:
                x1, y1, x2, y2, confidence, cls = box[:6]
                if confidence >= conf:
                    x1, y1, x2, y2 = map(int, (x1, y1, x2, y2))
                    cropped = img[y1:y2, x1:x2]
                    if cropped.size > 0:
                        save_path = os.path.join(save_dir, f"receipt_{label}_{img_counter}.jpg")
                        cv2.imwrite(save_path, cropped)
                        print(f"[INFO] Saved {save_path}")
                        return cropped
        return None

    # Try original image first
    cropped = try_detect(image, "original")
    if cropped is not None:
        return cropped

    print("[INFO] No detection in original. Trying shifts and zooms...")

    shifts = [(-50, 0), (50, 0), (0, -50), (0, 50)]  # left, right, up, down
    zoom_factors = [1.05, 1.1]  # slight zoom in

    # Try shifted images
    for dx, dy in shifts:
        M = np.float32([[1, 0, dx], [0, 1, dy]])
        shifted_img = cv2.warpAffine(image, M, (img_w, img_h))
        cropped = try_detect(shifted_img, f"shift_{dx}_{dy}")
        if cropped is not None:
            return cropped

    # Try zoomed-in images
    for zoom in zoom_factors:
        new_w = int(img_w * zoom)
        new_h = int(img_h * zoom)
        resized = cv2.resize(image, (new_w, new_h))

        # Crop center of resized image to original size
        x_start = (new_w - img_w) // 2
        y_start = (new_h - img_h) // 2
        zoomed_img = resized[y_start:y_start + img_h, x_start:x_start + img_w]

        cropped = try_detect(zoomed_img, f"zoom_{zoom}")
        if cropped is not None:
            return cropped

    print("[INFO] No receipts detected after shift and zoom attempts.")
    return None


def Start_yolo(img, pytes):
    try:

        print("Starting YOLO...")

        # Load the trained model
        model_path = "C:\\Porjects\\My-Fridge\\yolov8_training\\receipt_detector13\\weights\\best.pt"
        model = load_yolo_model(model_path)

        img = shift_and_zoom_detect(model, img)

        json = test.OCR(img, pytes)

        if img is not None:
            print("Image processed successfully.")
            print(f"JSON: {json}")
            return json
        else:
            return None

    except Exception as e:
        print(f"An error occurred: {e}")
        return e

def model_test(model):
    results = model.val(
        data="data.yaml",  # Must point to a valid YOLO data.yaml
        split="test",  # Ensure the 'test' set is defined in the YAML file
        save=True,
        save_txt=True,
        save_hybrid=True,
        save_conf=True
    )

    # Print metrics
    # print(f"Precision: {results.box.precision.mean():.3f}")
    # print(f"Recall: {results.box.recall.mean():.3f}")
    # print(f"mAP@0.5: {results.box.map50:.3f}")
    # print(f"mAP@0.5:0.95: {results.box.map:.3f}")

    class_names = model.names

    for i, iou in enumerate(results.box.iou):
        print(f"Class: {class_names[i]}, IoU: {iou:.3f}")


if __name__ == '__main__':

    # image_path = 'C:\\Projects\\MyFridge\\datasets\\Data\\valid\\images'
    # save_path = 'C:\\Projects\\MyFridge\\datasets\\Data\\valid\\processed'
#
    # if not os.path.isfile(image_path):
    #     print(f"Error: {image_path} is not a file")
#
    # image_preprocessing(image_path, save_path)

    # Train the YOLOv8 model
    # train_yolo()
    #TODO: add precision, recall and intersection over Union

    # Load the trained model
    model_path = "C:\\Porjects\\My-Fridge\\yolov8_training\\receipt_detector13\\weights\\best.pt"
    model = load_yolo_model(model_path)

    # Preform Model Evaluation

    model_test(model)

    # # Perform live receipt detection
    # ip_camera_url = "http://192.168.1.128:8080/video"  # Replace with your phone’s actual IP
    # detect_receipts(model, source=ip_camera_url)  # Run detection using IP webcam