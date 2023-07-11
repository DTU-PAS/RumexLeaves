import cv2
import numpy as np

def split_to_image_list(data_folder, splits):
    img_list = []
    for s in splits:
        with open(f"{data_folder}/dataset_splits/{s}", "r+") as f:
            img_list = [line.replace('\n', '') for line in f.readlines()]
    return img_list

if __name__ == "__main__":
    dataset_folder = "/home/ronja/data/l515_imgs/RumexLeaves/Robot"
    data_fold = 1
    splits = ["random_train.txt"]
    image_list = split_to_image_list(dataset_folder, splits)

    mean = 0
    std = 0
    nb_samples = 0
    for img_file in image_list:
        img = cv2.imread(f"{dataset_folder}/{img_file}").astype(np.float32)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img /= 255
        data = np.reshape(img, (-1, img.shape[2]))
        mean += data.mean(0)
        std += data.std(0)
        nb_samples += 1

    mean /= nb_samples
    std /= nb_samples

    print(mean)
    print(std)