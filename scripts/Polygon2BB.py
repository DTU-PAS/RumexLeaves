import shutil

import cv2
import glob
import matplotlib.pyplot as plt
import os
import math
from math import atan2, cos, sin, sqrt, pi
import spatialmath as sm
import numpy as np
from annotation_converter.AnnotationConverter import AnnotationConverter
from annotation_converter.BoundingBox import BoundingBox
from annotation_converter.Annotation import Annotation


class Seg2BBGenerator:
    def __init__(self, img_files, path_to_annotation_file, oriented_bb=False, debug=False):
        self._img_files = img_files
        self._path_to_annotation_file = path_to_annotation_file
        self._oriented_bb = oriented_bb
        self._debug = debug

    def generate(self):
        for img_file in self._img_files:
            img_id = os.path.basename(img_file)
            print(f"Generating Bounding Boxes for {img_id}...")
            rumex_img = cv2.imread(img_file, cv2.IMREAD_UNCHANGED)
            annotation = AnnotationConverter.read_cvat_by_id(self._path_to_annotation_file, img_id)
            bb_annotation = Annotation(img_id, annotation.get_img_width(), annotation.get_img_height(), bb_list=[],
                                       polygon_list=[], polyline_list=[])
            for polyon_ann in annotation.polygon_list:
                if self._oriented_bb:
                    polyline_list = annotation.polyline_list
                    bb = self._extract_oriented_bb_for_polygon(polyon_ann, polyline_list, rumex_img.copy())
                else:
                    bb = self._extract_common_bb_for_polygon(polyon_ann, rumex_img)
                bb.set_label(polyon_ann.get_label())
                bb_annotation.bb_list.append(bb)
            AnnotationConverter.extend_cvat(bb_annotation, self._path_to_annotation_file)

            if self._debug:
                for bb in bb_annotation.bb_list:
                    if self._oriented_bb:
                        c1, c2, width, height = bb.get_x(), bb.get_y(), bb.get_width(), bb.get_height()
                        angle = bb.get_rotation()
                        rot_mat = sm.SE2.Rot(angle)
                        box_points = np.array([[-width / 2, -height / 2],
                                               [width / 2, -height / 2],
                                               [width / 2, height / 2],
                                               [-width / 2, height / 2]])
                        for i, box_point in enumerate(box_points):
                            box_point = sm.SE2(x=box_point[0], y=box_point[1])
                            box_point_trans = (rot_mat * box_point).t
                            box_points[i] = [box_point_trans[0] + c1, box_point_trans[1] + c2]
                        cv2.drawContours(rumex_img, [box_points.astype(int)], 0, (0, 0, 255), 2)
                    else:
                        c1, c2, width, height = bb.get_x() + bb.get_width() / 2, bb.get_y() + bb.get_height() / 2, bb.get_width(), bb.get_height()
                        box = cv2.boxPoints(((c1, c2), (width, height), bb.get_rotation()))
                        cv2.drawContours(rumex_img, [box.astype(int)], 0, (0, 0, 255), 2)

                plt.imshow(rumex_img)
                plt.show()

    def drawAxis(self, img, p_, q_, colour, scale):
        p = list(p_)
        q = list(q_)

        angle = atan2(p[1] - q[1], p[0] - q[0])  # angle in radians
        hypotenuse = sqrt((p[1] - q[1]) * (p[1] - q[1]) + (p[0] - q[0]) * (p[0] - q[0]))
        # Here we lengthen the arrow by a factor of scale
        q[0] = p[0] - scale * hypotenuse * cos(angle)
        q[1] = p[1] - scale * hypotenuse * sin(angle)
        cv2.line(img, (int(p[0]), int(p[1])), (int(q[0]), int(q[1])), colour, 5, cv2.LINE_AA)
        # create the arrow hooks
        p[0] = q[0] + 9 * cos(angle + pi / 4)
        p[1] = q[1] + 9 * sin(angle + pi / 4)
        cv2.line(img, (int(p[0]), int(p[1])), (int(q[0]), int(q[1])), colour, 5, cv2.LINE_AA)
        p[0] = q[0] + 9 * cos(angle - pi / 4)
        p[1] = q[1] + 9 * sin(angle - pi / 4)
        cv2.line(img, (int(p[0]), int(p[1])), (int(q[0]), int(q[1])), colour, 5, cv2.LINE_AA)

    def is_polyline_in_polygon(self, polyline, polygon_mask):
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

    def _extract_oriented_bb_for_polygon(self, polygon_ann, polyline_list, rumex_img, debug=False):
        polygon_pts = polygon_ann.get_polygon_points_as_array()

        # Create Leave Mask
        mask = np.zeros(rumex_img.shape[0:2], dtype=np.uint8)
        cv2.fillPoly(mask, [polygon_pts], (255))

        # Get corresponding polyline
        pl = None
        polyline_max = 0
        for polyline in polyline_list:
            if self.is_polyline_in_polygon(polyline, mask):
                pl = polyline
                break
        if not pl:
            print("No polyline found for polygon. Check the annotation correctness.")

        # Perform PCA analysis on polyline
        points = pl.get_polyline_points_as_array()
        mean = np.empty((0))
        mean, eigenvectors, eigenvalues = cv2.PCACompute2(points.astype(float), mean)

        cntr = (int(mean[0, 0]), int(mean[0, 1]))
        p1_1 = (cntr[0] + 0.02 * eigenvectors[0, 0] * eigenvalues[0, 0],
                cntr[1] + 0.02 * eigenvectors[0, 1] * eigenvalues[0, 0])
        p1_2 = (cntr[0] - 0.02 * eigenvectors[0, 0] * eigenvalues[0, 0],
                cntr[1] - 0.02 * eigenvectors[0, 1] * eigenvalues[0, 0])
        p2 = (cntr[0] - 0.02 * eigenvectors[1, 0] * eigenvalues[1, 0],
              cntr[1] - 0.02 * eigenvectors[1, 1] * eigenvalues[1, 0])

        leaf_tip = points[-1]
        d1 = math.sqrt(math.pow((p1_1[0] - leaf_tip[0]), 2) + math.pow((p1_1[1] - leaf_tip[1]), 2))
        d2 = math.sqrt(math.pow((p1_2[0] - leaf_tip[0]), 2) + math.pow((p1_2[1] - leaf_tip[1]), 2))
        if d1 < d2:
            p1 = p1_1
        else:
            p1 = p1_2
        angle = atan2(p1[1] - cntr[1], p1[0] - cntr[0])
        if angle < 0:
            angle = 2 * math.pi + angle

        # Get rotated bounding box with rotation towards leaf vein axis (pointing to leaf tip)
        enlarge_factor = 3
        el_matrix = np.zeros((mask.shape[0] * enlarge_factor, mask.shape[1] * enlarge_factor), dtype=np.uint8)
        el_matrix[int(mask.shape[0] * (enlarge_factor - 1) / 2):int(mask.shape[0] * (enlarge_factor + 1) / 2),
        int(mask.shape[1] * (enlarge_factor - 1) / 2):int(mask.shape[1] * (enlarge_factor + 1) / 2)] = mask

        el_cntr = (
            cntr[0] + mask.shape[1] * (enlarge_factor - 1) / 2, cntr[1] + mask.shape[0] * (enlarge_factor - 1) / 2)

        rot_mat = cv2.getRotationMatrix2D(el_cntr, angle * 180 / math.pi, 1.0)
        mask = cv2.warpAffine(el_matrix, rot_mat, el_matrix.shape[1::-1], flags=cv2.INTER_LINEAR)

        contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        x1, y1, width, height = cv2.boundingRect(contours[0])
        bb_center = (x1 + width / 2, y1 + height / 2)
        bb_points = np.array([[x1, y1], [x1 + width, y1], [x1 + width, y1 + height], [x1, y1 + height]])

        rot_mat = sm.SE2.Rot(angle)
        bb_center = sm.SE2(x=bb_center[0] - el_cntr[0], y=bb_center[1] - el_cntr[1])
        bb_center = (rot_mat * bb_center).t
        bb_center = (bb_center[0] + cntr[0], bb_center[1] + cntr[1])

        bb = BoundingBox("", bb_center[0], bb_center[1], width, height, angle)

        # Debugging
        if debug:
            # Draw polygon
            cv2.fillPoly(rumex_img, [polygon_pts], (255, 0, 0))

            # Draw bounding box
            cx, cy, width, height = bb.get_xywh()
            bb_points = np.array([[cx - width / 2, cy - height / 2],
                                  [cx + width / 2, cy - height / 2],
                                  [cx + width / 2, cy + height / 2],
                                  [cx - width / 2, cy + height / 2]])
            for i, bb_point in enumerate(bb_points):
                bb_point = sm.SE2(x=bb_point[0] - cx, y=bb_point[1] - cy)
                bb_point = (rot_mat * bb_point).t
                bb_point[0] += cx
                bb_point[1] += cy
                bb_points[i] = bb_point

            cv2.circle(rumex_img, (int(cx), int(cy)), 10, (0, 0, 255), -1)
            cv2.drawContours(rumex_img, [bb_points.astype(int)], 0, (255, 0, 0), 2)

            # Draw polyline
            points = pl.get_polyline_points_as_array()
            cv2.polylines(rumex_img, [points], 0, (100, 100, 0), 2)
            for p in points:
                cv2.circle(rumex_img, (p[0], p[1]), 5, (100, 100, 0), -1)

            # Draw the principal components
            self.drawAxis(rumex_img, cntr, p1, (0, 255, 0), 1)
            self.drawAxis(rumex_img, cntr, p2, (255, 255, 0), 10)

            # Show debug image
            plt.imshow(rumex_img)
            plt.show()
        return bb

    def _extract_common_bb_for_polygon(self, polygon_ann, rumex_img):
        polygon_pts = polygon_ann.get_polygon_points_as_array()
        rumex_img_tmp = rumex_img.copy()

        # Create Leave Mask
        mask = np.zeros(rumex_img.shape[0:2], dtype=np.uint8)
        cv2.fillPoly(mask, [polygon_pts], (255))
        contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        x1, y1, width, height = cv2.boundingRect(contours[0])
        bb = BoundingBox("", x1, y1, width, height, 0)

        return bb


if __name__ == "__main__":
    debug = False
    path_to_dataset = "/home/ronja/data/generated/RumexLeaves/2023-06-09T15-38-49_rumexleaves-ldm-vq-4_pretr1_3"
    sub_locations = glob.glob(f"{path_to_dataset}/*") #[f"{path_to_dataset}/iNaturalist"]
    ann_file_name = "annotations.xml"
    oriented_bb = True
    if not oriented_bb:
        ann_file_name_new = "annotations_bb.xml"
    else:
        ann_file_name_new = "annotations_oriented_bb.xml"

    for sub_location in sub_locations:
        ann_file = f"{sub_location}/images/{ann_file_name}"
        ann_file_new = f"{sub_location}/images/{ann_file_name_new}"
        shutil.copy(ann_file, ann_file_new)
        image_files = glob.glob(f"{sub_location}/*/*/*rgb*.png")
        image_files = glob.glob(f"{sub_location}/images/*.png")
        lcg = Seg2BBGenerator(image_files, ann_file_new, oriented_bb, debug)
        lcg.generate()
