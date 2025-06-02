import os
import cv2 as cv
import random
import shutil

# Copy images and create empty label files
def process_split(split_name, file_list):
    for filename in file_list:
        src_path = os.path.join(dataset_path, filename)
        dst_img_path = os.path.join(split_name, "images", filename)
        dst_lbl_path = os.path.join(split_name, "labels", filename.rsplit(".", 1)[0] + ".txt")

        # Copy image
        img = cv.imread(src_path, cv.IMREAD_GRAYSCALE)
        if img is not None:
            cv.imwrite(dst_img_path, img)

        # Create empty .txt label
        open(dst_lbl_path, "w").close()


if __name__ == '__main__':

    # Set your dataset path (e.g., COCO val2017)
    dataset_path = "val2017"
    extra_data_count = 2000
    train_ratio = 0.8
    val_ratio = 0.1
    test_ratio = 0.1
    random.seed(42)

    # Get all image file names
    all_images = [f for f in os.listdir(dataset_path) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
    print(len(all_images))
    print(all_images)

    # Randomly select extra_data_count images
    extra_data_index_list = random.sample(range(0, len(all_images)), extra_data_count)
    extra_data_name = [all_images[i] for i in extra_data_index_list]
    print(len(extra_data_name))
    print(extra_data_name)

    # Split into train, val, test
    random.shuffle(extra_data_name)
    train_end = int(train_ratio * extra_data_count)
    val_end = train_end + int(val_ratio * extra_data_count)
    train = extra_data_name[:train_end]
    val = extra_data_name[train_end:val_end]
    test = extra_data_name[val_end:]

    # Create folder structure
    splits = {"train": train, "val": val, "test": test}
    for split in splits:
        os.makedirs(f"{split}/images", exist_ok=True)
        os.makedirs(f"{split}/labels", exist_ok=True)

    # Process each split
    for split in splits:
        process_split(split, splits[split])

    print("Done! Dataset split into train/val/test with empty labels.")
