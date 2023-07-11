import shutil

import cv2
import glob
import os
import numpy as np
from annotation_converter.AnnotationConverter import AnnotationConverter

def main():
    dataset_path = "/home/ronja/data/l515_imgs/RumexLeaves/iNaturalist"
    output_path = "/home/ronja/data/l515_imgs/RumexLeaves/iNaturalist/segmentations"
    os.makedirs(output_path, exist_ok=True)
    img_files = glob.glob(f"{dataset_path}/*.jpg")
    annotation_file = f"{dataset_path}/annotations.xml"
    labels = {"leaf_blade": 1, "leaf_stem": 2}

    for img_file in img_files:
        img = cv2.imread(img_file)
        annotation = AnnotationConverter.read_cvat_by_id(annotation_file, os.path.basename(img_file))
        mask = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)

        polygons = annotation.get_polygons()
        for polygon in polygons:
            points = polygon.get_polygon_points_as_array()
            cv2.fillPoly(mask, [points], (labels["leaf_blade"]))
        
        polylines = annotation.get_polylines()
        for polyline in polylines:
            points = polyline.get_polyline_points_as_array()
            cv2.polylines(mask, [points], False, (labels["leaf_stem"]), 25)
        ouput_file_name = f"{output_path}/{os.path.basename(img_file).replace('.jpg', '.png')}"
        cv2.imwrite(ouput_file_name, mask)
        # fig, ax = plt.subplots(1, 2)
        # ax[0].imshow(img)
        # ax[1].imshow(mask)
        # plt.show()


if __name__ == '__main__':
    main()