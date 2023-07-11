import random
import cv2
import glob
import matplotlib.pyplot as plt
import os
import numpy as np

from annotation_converter.AnnotationConverter import AnnotationConverter


def display_overlay_of_inaturalist_images():
    dataset_path = "/home/ronja/data/l515_imgs/RumexLeaves/iNaturalist"
    predictions_path = "/home/ronja/log/center_leaf/multi_task2/iNat_test/multi_task2_iNat/debug/paper_imgs"
    output_path = "/home/ronja/data/l515_imgs/RumexLeaves/overlay_annotations"

    os.makedirs(output_path, exist_ok=True)
    img_files = glob.glob(f"{dataset_path}/*.jpg")
    annotation_file = f"{dataset_path}/annotations.xml"
    labels = {"leaf_blade": (0, 255, 255), "leaf_stem": (0, 0, 255)}

    # ToDO: Find diverse images for iNaturalist
    img_files = ["180.jpg", 
            "2044.jpg",
            "3824.jpg",
            "2819.jpg",
            "3791.jpg",
            "4150.jpg"]
    # img_files = ["1224.jpg"]
    h, w = 2048, 1536
    scale_percent = 20 # percent of original size
    width = int(w * scale_percent / 100)
    height = int(h * scale_percent / 100)
    dim = (width, height)

    width_factor = 2
    if predictions_path is not None:
        width_factor = 3
    grid = np.zeros((height * len(img_files), width * width_factor, 3), dtype=np.uint8)

    for i, img_file in enumerate(img_files):
        img_orig = cv2.imread(f"{dataset_path}/{img_file}")
        img = img_orig.copy()

        mask = np.zeros_like(img_orig)    
        annotation = AnnotationConverter.read_cvat_by_id(annotation_file, os.path.basename(img_file))
        polygons = annotation.get_polygons()
        for polygon in polygons:
            points = polygon.get_polygon_points_as_array()
            cv2.fillPoly(mask, [points], (labels["leaf_blade"]))
        
        img = cv2.addWeighted(mask, 0.5, img, 0.5, 0, img)
        
        polylines = annotation.get_polylines()
        for polyline in polylines:
            points = polyline.get_polyline_points_as_array()
            print(points)
            cv2.polylines(img, [points], False, labels["leaf_stem"], 10)
            for k, point in enumerate(points):
                thickness = 15
                if k in [0, 3, 7]:
                    thickness = 22
                cv2.circle(img, tuple(point), thickness, labels["leaf_stem"], -1)
        
        h, w = img_orig.shape[:2]
        if w > h:
            img_orig = cv2.rotate(img_orig, cv2.ROTATE_90_CLOCKWISE)
            img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
        
        img = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
        img_orig = cv2.resize(img_orig, dim, interpolation = cv2.INTER_AREA)
        grid[i * height:(i + 1) * height, 0:width] = img_orig
        grid[i * height:(i + 1) * height, width:2*width] = img
        if predictions_path is not None:
            pred_img = cv2.imread(f"{predictions_path}/{img_file}")
            if w > h:
                pred_img = cv2.resize(pred_img, (int(pred_img.shape[0] * w/pred_img.shape[1]), w), interpolation = cv2.INTER_AREA)
                
            else:
                pred_img = cv2.resize(pred_img, (h, int(pred_img.shape[1] * h/pred_img.shape[0])), interpolation = cv2.INTER_AREA)
            
            center = [pred_img.shape[0] / 2, pred_img.shape[0] / 2]
            x = int(center[1] - w/2)
            y = int(center[0] - h/2)
            pred_img = pred_img[y:y+h, x:x+w]

            if w > h:
                pred_img = cv2.rotate(pred_img, cv2.ROTATE_90_CLOCKWISE)
            pred_img = cv2.resize(pred_img, dim, interpolation = cv2.INTER_AREA)

            grid[i * height:(i + 1) * height, 2*width:3*width] = pred_img

    grid = cv2.rotate(grid, cv2.ROTATE_90_CLOCKWISE)
    ouput_file_name = f"{output_path}/iNaturalist_overlay.png"
    cv2.imwrite(ouput_file_name, grid)

    plt.imshow(grid)
    plt.show()

def display_overlay_of_robot_images():
    dataset_path = "/home/ronja/data/l515_imgs/RumexLeaves/Robot_cropped"
    predictions_path = "/home/ronja/log/center_leaf/multi_task2/robot_test/multi_task2_Robot/debug/paper_imgs"
    output_path = "/home/ronja/data/l515_imgs/RumexLeaves/overlay_annotations"
    os.makedirs(output_path, exist_ok=True)
    img_files = glob.glob(f"{dataset_path}/*/*/*/*_rgb_*.png")
    random.shuffle(img_files)
    img_files = img_files[0:6]

    img_files = [
        glob.glob(f"{dataset_path}/*/*/*/20210623_skaevinge_rgb_0_1624440580560785294.png")[0],
        glob.glob(f"{dataset_path}/*/*/*/20210623_skaevinge_rgb_0_1624440865063445330.png")[0],
        glob.glob(f"{dataset_path}/*/*/*/20210623_skaevinge_rgb_0_1624441162015172243.png")[0],
        glob.glob(f"{dataset_path}/*/*/*/20210806_stengard_rgb_0_1628243873297839165.png")[0],
        glob.glob(f"{dataset_path}/*/*/*/20210806_stengard_rgb_1_1628244536330368757.png")[0],
        glob.glob(f"{dataset_path}/*/*/*/20210908_skaevinge_rgb_0_1631091680458191395.png")[0],
        ]

    labels = {"leaf_blade": (0, 255, 255), "leaf_stem": (0, 0, 255)}

    crop = [[270, 1080], [320, 1130]]
    h, w = 512, 512
    dim = (w, h)

    width_scale = 3
    if predictions_path is not None:
        width_scale = 4

    grid = np.zeros((h * len(img_files), w *width_scale, 3), dtype=np.uint8)

    for i, img_file in enumerate(img_files):
        img_orig = cv2.imread(img_file)
        depth_orig = cv2.imread(img_file.replace("rgb", "depth").replace("Robot_cropped", "Robot"), cv2.IMREAD_ANYDEPTH)
        depth_orig = depth_orig[crop[0][0]:crop[0][1], crop[1][0]:crop[1][1]]

        img = img_orig.copy()

        mask = np.zeros_like(img_orig)
        annotation_file = f"{os.path.dirname(img_file)}/../../annotations.xml"
        annotation = AnnotationConverter.read_cvat_by_id(annotation_file, os.path.basename(img_file))
        polygons = annotation.get_polygons()
        for polygon in polygons:
            points = polygon.get_polygon_points_as_array()
            cv2.fillPoly(mask, [points], (labels["leaf_blade"]))
        
        img = cv2.addWeighted(mask, 0.5, img, 0.5, 0, img)
        
        polylines = annotation.get_polylines()
        for polyline in polylines:
            points = polyline.get_polyline_points_as_array()
            cv2.polylines(img, [points], False, labels["leaf_stem"], 4)
            for k, point in enumerate(points):
                thickness = 7
                if k in [0, 3, 7]:
                    thickness = 12
                cv2.circle(img, tuple(point), thickness, labels["leaf_stem"], -1)

        
        img = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
        img_orig = cv2.resize(img_orig, dim, interpolation = cv2.INTER_AREA)
        depth_orig = cv2.resize(depth_orig, dim, interpolation = cv2.INTER_AREA)
        depth_orig = depth_orig.astype(np.uint8)
        depth_orig = cv2.applyColorMap(depth_orig, cv2.COLORMAP_JET)
        grid[i * h:(i + 1) * h, 0:w] = img_orig
        grid[i * h:(i + 1) * h, w:2*w] = depth_orig
        grid[i * h:(i + 1) * h, 2*w:3*w] = img

        if predictions_path is not None:
            pred_img = cv2.imread(f"{predictions_path}/{os.path.basename(img_file)}")
            grid[i * h:(i + 1) * h, 3*w:4*w] = cv2.resize(pred_img, dim, interpolation = cv2.INTER_AREA)

    grid = cv2.rotate(grid, cv2.ROTATE_90_CLOCKWISE)
    ouput_file_name = f"{output_path}/Robot_overlay.png"
    cv2.imwrite(ouput_file_name, grid)

    plt.imshow(grid)
    plt.show()

def display_overlay_of_generated_images():
    dataset_path = "/home/ronja/data/generated/RumexLeaves/2023-06-09T11-09-41_rumexleaves-ldm-vq-4_pretr1_2/epoch=000099"
    predictions_path = None
    output_path = f"{os.path.dirname(dataset_path)}/overlay_annotations"

    os.makedirs(output_path, exist_ok=True)
    img_files = glob.glob(f"{dataset_path}/images/*.png")
    annotation_file = f"{dataset_path}/annotations.xml"
    labels = {"leaf_blade": (0, 255, 255), "leaf_stem": (0, 0, 255)}

    img_files = random.choices(img_files, k=6)

    width = 256
    height = 256
    dim = (width, height)


    width_factor = 2
    if predictions_path is not None:
        width_factor = 3
    grid = np.zeros((height * len(img_files), width * width_factor, 3), dtype=np.uint8)

    for i, img_file in enumerate(img_files):
        img_orig = cv2.imread(img_file)
        h, w = img_orig.shape[:2]

        img = img_orig.copy()

        annotation = AnnotationConverter.read_cvat_by_id(annotation_file, os.path.basename(img_file))
        mask = np.zeros_like(img)   
        polygons = annotation.get_polygons()
        for polygon in polygons:
            points = polygon.get_polygon_points_as_array()
            points = np.array(points).astype(np.int64)

            cv2.fillPoly(mask, [points], (labels["leaf_blade"]))
        
        polylines = annotation.get_polylines()
        for polyline in polylines:
            points = polyline.get_polyline_points_as_array()
            points = np.array(points).astype(np.int64)

            cv2.polylines(img, [points], False, labels["leaf_stem"], 2)
            for k, point in enumerate(points):
                thickness = 3
                if k in [0, 3, 7]:
                    thickness = 5
                cv2.circle(img, tuple(point), thickness, labels["leaf_stem"], -1)
        
        img = cv2.addWeighted(mask, 0.5, img, 0.5, 0, img)
        
        img = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
        img_orig = cv2.resize(img_orig, dim, interpolation = cv2.INTER_AREA)
        grid[i * height:(i + 1) * height, 0:width] = img_orig
        grid[i * height:(i + 1) * height, width:2*width] = img

        if predictions_path is not None:
            pred_img = cv2.imread(f"{predictions_path}/{img_file}")
            if w > h:
                pred_img = cv2.resize(pred_img, (int(pred_img.shape[0] * w/pred_img.shape[1]), w), interpolation = cv2.INTER_AREA)
                
            else:
                pred_img = cv2.resize(pred_img, (h, int(pred_img.shape[1] * h/pred_img.shape[0])), interpolation = cv2.INTER_AREA)
            
            center = [pred_img.shape[0] / 2, pred_img.shape[0] / 2]
            x = int(center[1] - w/2)
            y = int(center[0] - h/2)
            pred_img = pred_img[y:y+h, x:x+w]

            if w > h:
                pred_img = cv2.rotate(pred_img, cv2.ROTATE_90_CLOCKWISE)
            pred_img = cv2.resize(pred_img, dim, interpolation = cv2.INTER_AREA)

            grid[i * height:(i + 1) * height, 2*width:3*width] = pred_img

    grid = cv2.rotate(grid, cv2.ROTATE_90_CLOCKWISE)
    ouput_file_name = f"{output_path}/overlay.png"
    cv2.imwrite(ouput_file_name, grid)

    plt.imshow(grid)
    plt.show()

def main():
    display_overlay_of_generated_images()
    # display_overlay_of_robot_images()
    # display_overlay_of_inaturalist_images()

if __name__ == '__main__':
    main()