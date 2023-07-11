import albumentations as A
import argparse
import albumentations.pytorch as AP
import cv2
import matplotlib.pyplot as plt
import numpy as np
from rumex_leaves.data import RumexLeavesDataset

colors = [(255, 215, 0),
        (255, 69, 0)]

def tensor_to_rgb(img):
    img = img.permute(1, 2, 0)
    img = img.numpy()
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

def visualize_iNaturalist(data_folder, annotation_file, img_list, num_images):
    transform = A.Compose([
        AP.ToTensorV2(),
    ])

    dataset = RumexLeavesDataset(
        data_dir=data_folder,
        image_list=img_list,
        preproc=transform,
        annotation_file=annotation_file,
    )

    count = 0
    while count < num_images:
        i = np.random.randint(0, len(dataset))
        img, masks, img_info = dataset[i]

        img = tensor_to_rgb(img)

        mask_img = np.zeros_like(img)
        mask_img[masks[:, :, 0] == 1] = colors[0]
        mask_img[masks[:, :, 1] == 1] = colors[1]
        overlay = cv2.addWeighted(img, 0.5, mask_img, 0.5, 0)

        fig, ax = plt.subplots(1, 1, figsize=(16, 8))
        ax.imshow(overlay)
        plt.show()
        count += 1


if __name__ == "__main__":
    argparse = argparse.ArgumentParser()
    argparse.add_argument("--data_folder", type=str)
    argparse.add_argument("--num_images", type=int, default=1)
    argparse.add_argument("--datapoint_type", type=str, default="iNaturalist", choices=["RoboRumex", "iNaturalist"])

    args = argparse.parse_args()

    # Load the training data
    splits = ["random_train.txt"]
    img_list = []
    for s in splits:
        with open(f"{args.data_folder}/{args.datapoint_type}/dataset_splits/{s}", "r+") as f:
            img_list = [line.replace('\n', '') for line in f.readlines()]
        
        if args.datapoint_type == "iNaturalist":
            annotation_file = "annotations.json"
        elif args.datapoint_type == "RoboRumex":
            annotation_file = "../../annotations.xml"

        visualize_iNaturalist(f"{args.data_folder}/{args.datapoint_type}", annotation_file, img_list, args.num_images)
