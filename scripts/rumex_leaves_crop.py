import glob
import shutil
import os
import cv2
import matplotlib.pyplot as plt

from annotation_converter.AnnotationConverter import AnnotationConverter
from annotation_converter.BoundingBox import BoundingBox
from annotation_converter.Polyline import Polyline
from annotation_converter.Polygon import Polygon
from annotation_converter.Annotation import Annotation

def remap_points(points, crop):
    crop_height = crop[0][1] - crop[0][0]
    crop_width = crop[1][1] - crop[1][0]
    for i, point in enumerate(points):
        point[0] -= crop[1][0]
        point[1] -= crop[0][0]
        if point[0] < 0:
            point[0] = 0
        if point[1] < 0:
            point[1] = 0
        if point[0] >= crop_width:
            point[0] = crop_width - 1
        if point[1] >= crop_height:
            point[1] = crop_height - 1
        points[i] = point
    return points

def main():
    path_to_rumex_leaves = "/home/ronja/data/l515_imgs/RumexLeaves/Robot"
    output_dataset = "/home/ronja/data/l515_imgs/Robot_cropped"
    annotation_file_identifier = "annotations.xml"
    debug = False
    crop = [[270, 1080], [320, 1130]]
    crop_height = crop[0][1] - crop[0][0]
    crop_width = crop[1][1] - crop[1][0]

    shutil.copytree(path_to_rumex_leaves, output_dataset)

    sub_locations = glob.glob(f"{output_dataset}/20*")
    
    for sub_location in sub_locations:
        annotation_file = f"{sub_location}/{annotation_file_identifier}"
        img_files = glob.glob(f"{sub_location}/*/imgs/*rgb*.png")

        remapped_annotations = []
        for img_file in img_files:
            # Crop image
            img_id = os.path.basename(img_file)
            img = cv2.imread(img_file)
            img_depth = cv2.imread(img_file.replace("_rgb_", "_depth_"))
            cropped_img = img[crop[0][0]:crop[0][1], crop[1][0]:crop[1][1]]
            cropped_img_depth = img_depth[crop[0][0]:crop[0][1], crop[1][0]:crop[1][1]]
            cv2.imwrite(img_file, cropped_img)
            cv2.imwrite(img_file.replace("_rgb_", "_depth_"), cropped_img_depth)

            # Adapt annotations to cropped image
            annotation = AnnotationConverter.read_cvat_by_id(annotation_file, img_id)

            if not annotation:
                os.remove(img_file)
                os.remove(img_file.replace("_rgb_", "_depth_"))
                continue

            remapped_annotation = Annotation(img_id, crop_width, crop_height, bb_list=[], polygon_list=[], ellipse_list=[], polyline_list=[])
            # Remap annotations
            for polygon in annotation.polygon_list:
                points = polygon.get_polygon_points_as_array()
                points = remap_points(points, crop)
                new_pol = Polygon(polygon.get_label())
                new_pol.set_polygon_points_as_array(points)
                remapped_annotation.add_polygon(new_pol)

            for polyline in annotation.polyline_list:
                points = polyline.get_polyline_points_as_array()
                points = remap_points(points, crop)
                new_poly = Polyline(polyline.get_label())
                new_poly.set_polyline_points_as_array(points)
                remapped_annotation.add_polyline(new_poly)

            for bb in annotation.bb_list:
                x, y, w, h = bb.get_xywh()
                new_bb = BoundingBox(x - crop[1][0], y - crop[0][0], w, h)
                remapped_annotation.add_bounding_box(new_bb)
            remapped_annotations.append(remapped_annotation)
            # Debug
            if debug:
                cropped_img_debug = cropped_img.copy()
                for polygon in remapped_annotation.polygon_list:
                    points = polygon.get_polygon_points_as_array()
                    cv2.polylines(cropped_img_debug, [points], True, (0, 0, 255), 2)
                for polyline in remapped_annotation.polyline_list:
                    points = polyline.get_polyline_points_as_array()
                    cv2.polylines(cropped_img_debug, [points], False, (255, 0, 0), 2)
                for bb in remapped_annotation.bb_list:
                    x, y, w, h = bb.get_xywh()
                    cv2.rectangle(cropped_img_debug, (x, y), (x + w, y + h), (0, 255, 0), 2)
                plt.imshow(cropped_img_debug)
                plt.show()
        AnnotationConverter.write_cvat(remapped_annotations, annotation_file)


if __name__ == "__main__":
    main()