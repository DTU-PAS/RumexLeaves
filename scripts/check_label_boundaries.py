import glob
import os 
from annotation_converter.AnnotationConverter import AnnotationConverter

def main():
    dataset_path = "/home/ronja/data/l515_imgs/RumexLeaves/iNaturalist"
    annotation_file = f"{dataset_path}/annotations.xml"
    annotations = AnnotationConverter.read_cvat_all(annotation_file)
    for j, annotation in enumerate(annotations):
        img_id = annotation.get_image_name()
        polylines = annotation.get_polylines()
        img_width = annotation.get_img_width()
        img_height = annotation.get_img_height()
        for k, polyline in enumerate(polylines):
            points = polyline.get_polyline_points_as_array()
            for i, point in enumerate(points):
                if point[0] == img_width:
                    print(f"{img_id}, {i}, {point}")
                    polyline.points["x"][i] -= 1
                if point[1] == img_height:
                    print(f"{img_id}, {i}, {point}")
                    polyline.points["y"][i] -= 1
            polylines[k] = polyline
        annotation.polyline_list = polylines
        annotations[j] = annotation
    AnnotationConverter.write_cvat(annotations, annotation_file)
        
if __name__ == '__main__':
    main()