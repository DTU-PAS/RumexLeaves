
import glob
import os
import numpy as np
from annotation_converter.AnnotationConverter import AnnotationConverter
from annotation_converter.BoundingBox import BoundingBox
from annotation_converter.Annotation import Annotation

def get_num_leaves(img_files, relative_annotation_file):
    num_leaves_w_stem = 0
    num_leaves_wo_stem = 0

    for img_file in img_files:
        annotation_file = f"{os.path.dirname(img_file)}/{relative_annotation_file}"
        annotation = AnnotationConverter.read_cvat_by_id(annotation_file, os.path.basename(img_file))
        polylines = annotation.get_polylines()

        for polyline in polylines:
            points = polyline.get_polyline_points_as_array()
            if len(points) == 8:
                num_leaves_w_stem += 1
            else:
                num_leaves_wo_stem += 1
    return num_leaves_w_stem, num_leaves_wo_stem

def main():
    path_to_rumex_leaves = "/home/ronja/data/l515_imgs/RumexLeaves"
    path_to_robot_data = f"{path_to_rumex_leaves}/Robot"
    path_to_iNat_data = f"{path_to_rumex_leaves}/iNaturalist"

    robot_img_files = glob.glob(f"{path_to_robot_data}/*/*/*/*_rgb_*.png")
    inat_img_files = glob.glob(f"{path_to_iNat_data}/*.jpg")

    rob_num_leaves_w_stem, rob_num_leaves_wo_stem = get_num_leaves(robot_img_files, "../../annotations.xml")
    iNat_num_leaves_w_stem, iNat_num_leaves_wo_stem = get_num_leaves(inat_img_files, "annotations.xml")

    stats = {"robot": {"num_imgs": len(robot_img_files),
                       "num_leaves_w_stem": rob_num_leaves_w_stem,
                        "num_leaves_wo_stem": rob_num_leaves_wo_stem},
            "iNat": {"num_imgs": len(inat_img_files),
                     "num_leaves_w_stem": iNat_num_leaves_w_stem,
                     "num_leaves_wo_stem": iNat_num_leaves_wo_stem},
            "total": {"num_imgs": len(robot_img_files) + len(inat_img_files),
                      "num_leaves_w_stem": rob_num_leaves_w_stem + iNat_num_leaves_w_stem,
                      "num_leaves_wo_stem": rob_num_leaves_wo_stem + iNat_num_leaves_wo_stem}}
    
    print(stats)


if __name__ == '__main__':
    main()