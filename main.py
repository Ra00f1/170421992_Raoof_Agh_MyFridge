from ultralytics import YOLO
import cv2
import torch
import numpy as np
import os


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


if __name__ == '__main__':

    # image_path = 'C:\\Projects\\MyFridge\\datasets\\Data\\valid\\images'
    # save_path = 'C:\\Projects\\MyFridge\\datasets\\Data\\valid\\processed'
#
    # if not os.path.isfile(image_path):
    #     print(f"Error: {image_path} is not a file")
#
    # image_preprocessing(image_path, save_path)

    # Train the YOLOv8 model
    train_yolo()
    #TODO: add precision, recall and intersection over Union

    # Load the trained model
    # model_path = "C:\\Porjects\\MyFridgeFinal\\yolov8_training\\receipt_detector15\\weights\\best.pt"
    # model = load_yolo_model(model_path)
#
    # # Perform live receipt detection
    # ip_camera_url = "http://192.168.1.128:8080/video"  # Replace with your phone’s actual IP
    # detect_receipts(model, source=ip_camera_url)  # Run detection using IP webcam