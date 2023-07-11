from annotation_converter.AnnotationConverter import AnnotationConverter
import random
import numpy as np
import cv2
import math
import matplotlib.pyplot as plt
from copy import deepcopy

def is_polyline_in_polygon(polyline, polygon_mask):
    points = polyline.get_polyline_points_as_array()
    counter = 0
    for point in points:
        point[0] = min(point[0], polygon_mask.shape[1] - 1)
        point[1] = min(point[1], polygon_mask.shape[0] - 1)
        try:
            if polygon_mask[int(point[1]), int(point[0])] == 255:
                counter += 1
        except Exception as e:
            print(e)
            print(polygon_mask.shape)
            print(point)
    if counter >= 4:
        return True
    return False

# ToDo: Detect occlusion and recalculate corresponding polygon and polyline
def main():
    annotation_file = "/home/ronja/data/l515_imgs/RumexLeaves/iNaturalist/annotations.xml"
    new_annotation_file = "/home/ronja/data/l515_imgs/RumexLeaves/iNaturalist/annotations_whole_leafs.xml"
    whole_leaf_annotations = []
    annotations = AnnotationConverter.read_cvat_all(annotation_file)
    for i, annotation in enumerate(annotations[100:120]):
        print(annotation.image_name)
        annotation_new = deepcopy(annotation)
        annotation_new.polygon_list = []
        annotation_new.polyline_list = []

        for polygon_ann in annotation.polygon_list:
            polygon_pts = polygon_ann.get_polygon_points_as_array()

            # Create Leave Mask
            mask = np.zeros((int(annotation.img_height), int(annotation.img_width)), dtype=np.uint8)
            cv2.fillPoly(mask, [polygon_pts], (255))

            # Get corresponding polyline
            pl = None
            polyline_max = 0
            for polyline in annotation.polyline_list:
                if is_polyline_in_polygon(polyline, mask):
                    pl = polyline
                    break
            if not pl:
                print("No polyline found for polygon. Check the annotation correctness.")
                break

            img = cv2.imread(f"/home/ronja/data/l515_imgs/RumexLeaves/iNaturalist/{annotation.image_name}")
            cv2.polylines(img, [polygon_pts], True, (0, 255, 0), 2)

            # plt.imshow(img)

            # plt.show()
            cv2.namedWindow('img', cv2.WINDOW_NORMAL)
            cv2.imshow("img", img)
            k = cv2.waitKey(10000)
            if k == 27:
                break
            elif k == 115:
                annotation_new.polygon_list.append(polygon_ann)
                annotation_new.polyline_list.append(pl)
            else:
                continue
        if k == 27:
            break
        AnnotationConverter.extend_cvat(annotation_new, new_annotation_file)
            
if __name__ == "__main__":
    main()