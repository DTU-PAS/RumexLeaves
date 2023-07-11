#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import cv2
import numpy as np
import os
from torch.utils.data import Dataset

from annotation_converter.AnnotationConverter import AnnotationConverter


class RumexLeavesDataset(Dataset):

    """
    RumexLeaves Dataset

    Args:
        data_dir (string): filepath to RumexLeaves folder.
        image_list (list(string)): list of img ids to consider
        preproc (callable, optional): transformation to perform on the
            input image
    """

    def __init__(
        self,
        data_dir,
        image_list,
        preproc=None,
        annotation_file="annotations.xml"
    ):
        super().__init__()
        self.data_dir = data_dir
        self.image_list = image_list
        self.preproc = preproc
        self.annotation_file = annotation_file
        self._classes = ["leaf_blade", "leaf_stem"]

    def __len__(self):
        return len(self.image_list)

    def _get_img_ids(self):
        ids = []
        for img_path in self.image_list:
            ids.append(os.path.basename(img_path))
        return ids


    def _load_anno_from_ids(self, id_):
        annotation_file = f"{self.data_dir}/{os.path.dirname(id_)}/{self.annotation_file}"
        img_id = os.path.basename(id_)
        img_annotations = AnnotationConverter.read_cvat_by_id(annotation_file, img_id)
        img_info = {}
        img_info["img_height"] = img_annotations.get_img_height()
        img_info["img_width"] = img_annotations.get_img_width()
        img_info["file_name"] = img_id
       
        all_polygons = img_annotations.get_polygons()
        all_polylines = img_annotations.get_polylines()

        return {"polygons": all_polygons, "polylines": all_polylines, "img_info": img_info}

    def load_anno(self, index):
        return self.annotations[index]

    def load_image(self, id_):
        img_file = os.path.join(self.data_dir, id_)

        img = cv2.imread(img_file)
        assert img is not None

        return img

    def __getitem__(self, index):
        id_ = self.image_list[index]
        ann = self._load_anno_from_ids(id_) #self.load_anno(index)
        img = self.load_image(id_)

        # Let's generate the mask from the polygons and polylines
        mask = np.zeros((img.shape[0], img.shape[1], len(self._classes)), dtype=np.uint8)
        polygons = ann["polygons"]
        for pol in polygons:
            pol_points = pol.get_polygon_points_as_array()
            label = pol.get_label()
            label_int = self._classes.index(label)
            color = [0, 0]
            color[label_int] = 1
            cv2.fillPoly(mask, [np.array(pol_points)], tuple(color))

        polylines = ann["polylines"]
        for pol in polylines:
            pol_points = pol.get_polyline_points_as_array()
            label = pol.get_label()
            label_int = self._classes.index(label)
            color = [0, 0]
            color[label_int] = 1
            cv2.polylines(mask, [np.array(pol_points)], False, tuple(color), 10)

        if self.preproc:
            transformed = self.preproc(image=img, mask=mask)
            img = transformed["image"]
            mask = transformed["mask"]

        return img, mask, ann["img_info"]

