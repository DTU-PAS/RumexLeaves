import cv2
import glob
import os

import matplotlib.pyplot as plt
import numpy as np
from annotation_converter.AnnotationConverter import AnnotationConverter


def num_kp_in_polygon(polyline, polygon_mask):
    w, h = polygon_mask.shape
    points = polyline.get_polyline_points_as_array()
    counter = 0
    for point in points:
        if point[0] < 0:
            point[0] = 0
        if point[0] >= h:
            point[0] = h - 1
        if point[1] < 0:
            point[1] = 0
        if point[1] >= w:
            point[1] = w - 1
        if polygon_mask[int(point[1]), int(point[0])] == 255:
            counter += 1
    return counter


def check_vein_blade_accordance(polygon_list, poly_list):
    return len(polygon_list), len(poly_list)


def check_number_of_leaf_kp(img, polygon_list, polyline_list):
    counter = 0
    for polygon_ann in polygon_list:
        mask = np.zeros(img.shape[0:2], dtype=np.uint8)
        cv2.fillPoly(mask, [polygon_ann.get_polygon_points_as_array()], (255))
        polyline_exist = False
        for polyline_ann in polyline_list:
            kp_in_polygon = num_kp_in_polygon(polyline_ann, mask)
            if kp_in_polygon == 5:
                polyline_exist = True
                break

        if not polyline_exist:
            # plt.imshow(mask)
            # plt.show()
            counter += 1
    return counter


def check_number_of_total_kp(polyline_list):
    counter = 0
    for poly_ann in polyline_list:
        if not (len(poly_ann.get_polyline_points_as_array()) == 5 or len(poly_ann.get_polyline_points_as_array()) == 8):
            counter += 1
    return counter


def main():
    # requirements
    path_to_rumex_leaves = "/home/ronja/data/l515_imgs/RumexLeaves/iNaturalist"
    annotation_file_relative_to_img = "annotations.xml"
    img_files = glob.glob(f"{path_to_rumex_leaves}/task_700/*.png")
    img_files.sort()


    num_images = 0

    for img_file in img_files:
        annotation_file = f"{os.path.dirname(img_file)}/{annotation_file_relative_to_img}"
        img_id = os.path.basename(img_file)
        img = cv2.imread(img_file)
        annotation = AnnotationConverter.read_cvat_by_id(annotation_file, img_id)
        if not annotation:
            # remove image
            os.remove(img_file)
            print(f"Annotation for {img_id} not found")
            continue
        num_images += 1
        polygon_list = annotation.get_polygons()
        poly_list = annotation.get_polylines()
        num_blade, num_stem = check_vein_blade_accordance(polygon_list, poly_list)
        if num_blade != num_stem :
            print(f"Image {img_id}: Number of veins/stems ({num_stem}) and blades ({num_blade}) are not equal.")
        num_of_incorrect_leaf_kps = check_number_of_leaf_kp(img, polygon_list, poly_list)
        if num_of_incorrect_leaf_kps > 0:
            print(f"Image {img_id}: {num_of_incorrect_leaf_kps} leaves with the incorrect number of keypoints in blade.")
        num_of_incorrect_num_kps_total = check_number_of_total_kp(poly_list)
        if num_of_incorrect_num_kps_total > 0 :
            print(f"Image {img_id}: Incorrect number of keypoints in {num_of_incorrect_num_kps_total} of veins/stems (should be either 5 or 8)")
    print(f"Number of images in dataset: {num_images}")



if __name__ == "__main__":
    main()
