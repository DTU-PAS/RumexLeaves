import glob
import random
import os
import json

def write_split(filename, img_list):
    with open(filename, 'w') as f:
        for item in img_list:
            f.write("%s\n" % item)

def split_to_img_list(data_folder, split_list):
    img_list = []
    with open(f"{data_folder}/dataset_splits/{split_list}", "r+") as f:
        img_list = [line.replace("\n", "") for line in f.readlines()]
    return img_list

def main_robot():
    rumex_leave_dataset = "/home/ronja/data/l515_imgs/RumexLeaves/Robot"
    img_files = glob.glob(f"{rumex_leave_dataset}/*/*/*/*_rgb_*.png")
    random.shuffle(img_files)
    train_split = 0.7
    val_split = 0.15
    train_imgs = img_files[:int(len(img_files) * train_split)]
    val_imgs = img_files[int(len(img_files) * train_split):int(len(img_files) * (train_split + val_split))]
    test_imgs = img_files[int(len(img_files) * (train_split + val_split)):]

    train_imgs = [f.replace(f"{rumex_leave_dataset}/", "") for f in train_imgs]
    val_imgs = [f.replace(f"{rumex_leave_dataset}/", "") for f in val_imgs]
    test_imgs = [f.replace(f"{rumex_leave_dataset}/", "") for f in test_imgs]

    write_split(f"{rumex_leave_dataset}/dataset_splits/random_train.txt", train_imgs)
    write_split(f"{rumex_leave_dataset}/dataset_splits/random_val.txt", val_imgs)
    write_split(f"{rumex_leave_dataset}/dataset_splits/random_test.txt", test_imgs)

def main_iNat():
    rumex_leave_dataset = "/home/ronja/data/l515_imgs/RumexLeaves/iNaturalist"
    rumex_leave_dataset = "/home/ronja/data/generated/RumexLeaves/composed_annotations"
    img_files = glob.glob(f"{rumex_leave_dataset}/*.png")
    random.shuffle(img_files)
    train_split = 1.0
    val_split = 0.15
    train_imgs = img_files[:int(len(img_files) * train_split)]
    val_imgs = img_files[int(len(img_files) * train_split):int(len(img_files) * (train_split + val_split))]
    test_imgs = img_files[int(len(img_files) * (train_split + val_split)):]

    train_imgs = [f.replace(f"{rumex_leave_dataset}/", "") for f in train_imgs]
    val_imgs = [f.replace(f"{rumex_leave_dataset}/", "") for f in val_imgs]
    test_imgs = [f.replace(f"{rumex_leave_dataset}/", "") for f in test_imgs]

    write_split(f"{rumex_leave_dataset}/dataset_splits/random_train.txt", train_imgs)
    write_split(f"{rumex_leave_dataset}/dataset_splits/random_val.txt", val_imgs)
    write_split(f"{rumex_leave_dataset}/dataset_splits/random_test.txt", test_imgs)

def main_composed():
    rumex_leave_dataset = "/home/ronja/data/generated/RumexLeaves/composed_annotations"
    img_files = glob.glob(f"{rumex_leave_dataset}/*.png")
    img_files = [f.replace(".png", ".jpg") for f in img_files]
    random.shuffle(img_files)
    train_split = 1.0
    val_split = 0.15
    train_imgs = img_files[:int(len(img_files) * train_split)]
    val_imgs = img_files[int(len(img_files) * train_split):int(len(img_files) * (train_split + val_split))]
    test_imgs = img_files[int(len(img_files) * (train_split + val_split)):]

    train_imgs = [f.replace(f"{rumex_leave_dataset}/", "") for f in train_imgs]
    val_imgs = [f.replace(f"{rumex_leave_dataset}/", "") for f in val_imgs]
    test_imgs = [f.replace(f"{rumex_leave_dataset}/", "") for f in test_imgs]

    os.makedirs(f"{rumex_leave_dataset}/dataset_splits", exist_ok=True)

    write_split(f"{rumex_leave_dataset}/dataset_splits/random_train.txt", train_imgs)
    write_split(f"{rumex_leave_dataset}/dataset_splits/random_val.txt", val_imgs)
    write_split(f"{rumex_leave_dataset}/dataset_splits/random_test.txt", test_imgs)


def main_gen():
    real_image_data_folder = "/home/ronja/data/l515_imgs/RumexLeaves/iNaturalist"
    real_image_train_list_path = f"{real_image_data_folder}/dataset_splits/random_train.txt"

    rel_gen_image_folder =  [
        "../../../generated/RumexLeaves/2023-06-09T15-38-49_rumexleaves-ldm-vq-4_pretr1_3/epoch=000099/images",
        "../../../generated/RumexLeaves/2023-06-09T15-38-49_rumexleaves-ldm-vq-4_pretr1_3/epoch=000149/images",
        "../../../generated/RumexLeaves/2023-06-09T15-38-49_rumexleaves-ldm-vq-4_pretr1_3/epoch=000199/images",
    ]

    with open(real_image_train_list_path, "r") as f:
        real_image_train_list = f.read().splitlines()
    
    fake_image_list = []
    for folder in rel_gen_image_folder:
        fake_image_list += glob.glob(f"{real_image_data_folder}/{folder}/*.png")
    fake_image_list = [f.replace(f"{real_image_data_folder}/", "") for f in fake_image_list]
    num_gen_images = len(fake_image_list)

    all_train_images = real_image_train_list + fake_image_list
    random.shuffle(all_train_images)

    os.makedirs(f"{real_image_data_folder}/dataset_splits", exist_ok=True)
    write_split(f"{real_image_data_folder}/dataset_splits/{os.path.basename(real_image_train_list_path).replace('.txt', '')}_plus_{num_gen_images}_gen.txt", all_train_images)


def fraction_train():
    image_data_folder = "/home/ronja/data/l515_imgs/RumexLeaves/iNaturalist"
    image_train_list_path = f"{image_data_folder}/dataset_splits/random_train.txt"

    fraction = 0.8
    with open(image_train_list_path, "r") as f:
        image_train_list = f.read().splitlines()
    
    sub_train_list = image_train_list[:int(len(image_train_list) * fraction)]

    os.makedirs(f"{image_data_folder}/dataset_splits", exist_ok=True)
    write_split(f"{image_data_folder}/dataset_splits/{os.path.basename(image_train_list_path).replace('.txt', '')}_frac_{fraction}.txt", sub_train_list)



if __name__ == "__main__":
    fraction_train()