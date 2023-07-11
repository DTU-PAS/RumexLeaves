from annotation_converter.AnnotationConverter import AnnotationConverter
from annotation_converter.Annotation import Annotation
from annotation_converter.Polygon import Polygon
from annotation_converter.Polyline import Polyline
import random
import numpy as np
import cv2
import math
import matplotlib.pyplot as plt
import os

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

def points_to_default_position(polygon, polyline, plant_pos):
    # Perform PCA analysis on polyline
    points = polyline.get_polyline_points_as_array()
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
    angle = math.atan2(p1[1] - cntr[1], p1[0] - cntr[0])
    if angle < 0:
        angle = 2 * math.pi + angle
    
    M = cv2.getRotationMatrix2D((int(points[0][0]), int(points[0][1])),angle*180/math.pi,1)
    rotated_polygon_points = rotate_points(polygon.get_polygon_points_as_array(), M)
    rotated_polyline_points = rotate_points(points, M)

    diff = plant_pos - points[0]
    rotated_polygon_points[:, 0] += diff[0]
    rotated_polygon_points[:, 1] += diff[1]
    rotated_polyline_points[:, 0] += diff[0]
    rotated_polyline_points[:, 1] += diff[1]

    return rotated_polygon_points, rotated_polyline_points

def rotate_points(points, M):
    ones = np.ones(shape=(len(points), 1))
    points_ones = np.hstack([points, ones])
    rotated_points = M.dot(points_ones.T).T
    rotated_points = rotated_points.astype(int)
    return rotated_points


# ToDo: Make sure spacing is equal and order is correct.
def get_reduced_polyline(polyline, polygon_mask, num_polyline_points=5):
    polyline_mask = np.zeros_like(polygon_mask)
    cv2.polylines(polyline_mask, [polyline], False, (255), 1)

    # Find the pixels with values of 150 and neighbors of 0 or 255
    overlay_mask = np.logical_and(polyline_mask == 255, polygon_mask == 0)

    # Set the selected pixels to 0
    polyline_mask[overlay_mask] = 0

    # Find the contours of the line
    contours, _ = cv2.findContours(polyline_mask.astype("uint8"), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Ensure there is only one contour
    if len(contours) == 0:
        return None
    if len(contours) > 2:
        print("Error: More than two contour found.")
        return None

    # Approximate the contour to a polyline
    # contours_new = []
    # for i, contour in enumerate(contours):
    #     epsilon = 0.0000001 * cv2.arcLength(contour, True)
    #     contours_new.append(cv2.approxPolyDP(contour, epsilon, True))
    # contours = contours_new
    polyline_cont = np.vstack(contours)
    if len(polyline_cont) <= 1:
        return None
    
    polyline_cont = sort_points_by_distance(polyline_cont.squeeze(), polyline[0])

    # Calculate the total length of the approximated curve
    total_length = cv2.arcLength(polyline_cont, True)

    # Calculate the spacing between each sampled point
    spacing = total_length / (num_polyline_points -1)

    # Sample points along the approximated curve
    points = []
    current_length = 0.0
    idx = 0

    for i in range(num_polyline_points):
        while cv2.arcLength(polyline_cont[:idx + 1], True) < current_length:
            idx += 1

        pt = polyline_cont[idx]
        points.append(pt)
        current_length += spacing

    return np.array(points).squeeze()

def distance(point1, point2):
    """Calculate Euclidean distance between two points."""
    x1, y1 = point1
    x2, y2 = point2
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

def sort_points_by_distance(points, reference_point):
    """Sort a list of points based on their distance to a reference point."""
    return np.array(sorted(points, key=lambda point: distance(point, reference_point)))

def polyine_to_static_format(polyline):
    if len(polyline) == 5:
        polyline = np.vstack((polyline[0], polyline[0], polyline[0], polyline))
    return polyline

def get_longest_contour(contours):
    longest_contour = None
    longest_contour_length = 0
    for contour in contours:
        if len(contour) > longest_contour_length:
            longest_contour = contour
            longest_contour_length = len(contour)
    return longest_contour

def compose_plant(annotations, plant_pos):
    plant_size = random.choice([3, 5, 10, 15, 30])
    number_of_leaves = plant_size + random.randint(0, int(plant_size/3)) * random.choice([1, -1])
    # number_of_leaves = random.randint(2, 30) # 30) #2
    plant_density = random.randint(10, 30)
    polygon_list = []
    polyline_list = []
    for i in range(number_of_leaves):
        annotation = random.choice(annotations)
        polygon_ann = random.choice(annotation.polygon_list) # annotation.polygon_list[leaf_num[i]] #
        polygon_pts = polygon_ann.get_polygon_points_as_array()

        # Create Leave Mask
        mask = np.zeros((int(annotation.img_height), int(annotation.img_width)), dtype=np.uint8)
        cv2.fillPoly(mask, [polygon_pts], (255))

        # Get corresponding polyline
        pl = None
        for polyline in annotation.polyline_list:
            if is_polyline_in_polygon(polyline, mask):
                pl = polyline
                break
        if not pl:
            print("No polyline found for polygon. Check the annotation correctness.")

        polygon_points, polyline_points = points_to_default_position(polygon_ann, pl, plant_pos)

        # Leaf offset from plant center
        offset = random.randint(0, number_of_leaves*plant_density)
        polyline_points[:, 0] += offset
        polygon_points[:, 0] += offset

        # Leaf Rotation around plant center
        angle = random.randrange(0, 360) 
        M = cv2.getRotationMatrix2D(plant_pos,angle,1)
        rotated_polygon_points = rotate_points(polygon_points, M)
        rotated_polyline_points = rotate_points(polyline_points, M)

        polygon_list.append(rotated_polygon_points)
        polyline_list.append(polyine_to_static_format(rotated_polyline_points))
    return polygon_list, polyline_list

def recompute_occluded_leaf(polygon_mask_occluding, polygon_mask_occluded, polyline_occluded):
    reduced_polygon_mask2 = cv2.subtract(polygon_mask_occluding, polygon_mask_occluded)
    reduced_polygon_mask2[reduced_polygon_mask2 > 0] = 0
    reduced_polygon_mask2[reduced_polygon_mask2 < 0] = 255

    # Reduced Polygon
    contours, hierarchy = cv2.findContours(reduced_polygon_mask2.astype("uint8"), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if len(contours) != 1:
        return None, None
    polygon_occluded = get_longest_contour(contours)
    gamma = 0.001
    while polygon_occluded.shape[0] > 100:
        epsilon = gamma * cv2.arcLength(polygon_occluded, True)
        polygon_occluded = cv2.approxPolyDP(polygon_occluded, epsilon, True)
        gamma *= 2

    reduced_polygon_mask2 = np.zeros_like(polygon_mask_occluding)
    cv2.fillPoly(reduced_polygon_mask2, [polygon_occluded], (255))
    orig_leaf_size = sum(sum(polygon_mask_occluded == 255))
    reduced_leaf_size = sum(sum(reduced_polygon_mask2 == 255))

    stem_mask = ((polygon_mask_occluding + polygon_mask_occluded) - 255) * -1
    stem_mask[stem_mask < 0] = 0

    if reduced_leaf_size < 3000 or reduced_leaf_size / orig_leaf_size < 0.33:
        # Leaf is too small, remove it
        recalc_polyline = None
    elif reduced_polygon_mask2[polyline_occluded[3][1], polyline_occluded[3][0]] == 0:
        # Basal covered, remove stem
        recalc_polyline = polyline_occluded
        recalc_polyline_tmp = get_reduced_polyline(polyline_occluded[3:], reduced_polygon_mask2, 5)
        if recalc_polyline_tmp is not None:
            recalc_polyline[3:] = recalc_polyline_tmp
            recalc_polyline[0:3] = recalc_polyline[3]
        else:
            recalc_polyline = None
    else:
        recalc_polyline = polyline_occluded
        recalc_polyline_tmp = get_reduced_polyline(polyline_occluded[3:], reduced_polygon_mask2, 5)
        if recalc_polyline_tmp is not None:
            recalc_polyline[3:] = recalc_polyline_tmp
            recalc_polyline[0:3] = recalc_polyline[0:3]
        else:
            recalc_polyline = None
        # Stem occluded ?
        if recalc_polyline is not None and np.any(polyline_occluded[0] != polyline_occluded[1]) and (stem_mask[polyline_occluded[2][1], polyline_occluded[2][0]] == 0 or stem_mask[polyline_occluded[1][1], polyline_occluded[1][0]] == 0 or stem_mask[polyline_occluded[0][1], polyline_occluded[0][0]] == 0):
            recalc_polyline_tmp = get_reduced_polyline(polyline_occluded[:4], stem_mask, 4)
            if recalc_polyline_tmp is not None:
                recalc_polyline[0:3] = recalc_polyline_tmp[:3]
            else:
                recalc_polyline[0:3] = recalc_polyline[3]
    polyline_occluded = recalc_polyline
    return polygon_occluded, polyline_occluded

def handle_occulsions(polygon_list, polyline_list, plant_pos):
    # Find occulsions and recalculate polyline and polygon annotation
    for i, (polygon_points1, polyline_points1) in enumerate(zip(polygon_list, polyline_list)):
        if polyline_points1 is None: 
            continue
        polygon_mask1 = np.zeros((int(plant_pos[0]*2), int(plant_pos[1]*2)))
        cv2.fillPoly(polygon_mask1, [polygon_points1], (255))
        dist_to_plant_center1 = np.linalg.norm(plant_pos - polyline_points1[3])

        for j, (polygon_points2, polyline_points2) in enumerate(zip(polygon_list, polyline_list)):
            if i == j or polyline_points2 is None:
                continue
            polygon_mask2 = np.zeros_like(polygon_mask1)
            cv2.fillPoly(polygon_mask2, [polygon_points2], (255))

            polyline_mask2 = np.zeros_like(polygon_mask1)
            cv2.polylines(polyline_mask2, [polyline_points2], False, (255), 1)

            intersection_leaf_blade = cv2.bitwise_and(polygon_mask1, polygon_mask2)
            intersection_stem = cv2.bitwise_and(polygon_mask1, polyline_mask2)

            # If occlusion
            if np.any(intersection_leaf_blade) or np.any(intersection_stem):
                if np.ma.allequal(intersection_leaf_blade, polygon_mask1):
                    polygon_list[i], polyline_list[i] = None, None
                    continue
                dist_to_plant_center2 = np.linalg.norm(plant_pos - polyline_points2[3])
                if dist_to_plant_center2 == dist_to_plant_center1:
                    dist_to_plant_center2 = np.linalg.norm(plant_pos - polyline_points2[7])
                    dist_to_plant_center1 = np.linalg.norm(plant_pos - polyline_points1[7])
                if dist_to_plant_center2 > dist_to_plant_center1:
                    polygon_occluded, polyline_occluded = recompute_occluded_leaf(polygon_mask1, polygon_mask2, polyline_points2)
                    polygon_list[j], polyline_list[j] = polygon_occluded, polyline_occluded
                    
    return polygon_list, polyline_list

def sort_leaves_by_distance(polygon_list, polyline_list, plant_pos):
    dist = [np.linalg.norm(plant_pos - polyline[3]) for polyline in polyline_list]
    sorted_indices = np.argsort(dist)
    polygon_list = [polygon_list[i] for i in sorted_indices]
    polyline_list = [polyline_list[i] for i in sorted_indices]
    return polygon_list, polyline_list

def crop_image(old_size, new_size, polygon_list, polyline_list):
    enlarged_image = np.zeros(old_size)
    enlarged_image[0:int((enlarged_image.shape[1] - new_size[1])/2), :] = 255
    enlarged_image[-int((enlarged_image.shape[1] - new_size[1])/2):-1, :] = 255
    enlarged_image[:, 0:int((enlarged_image.shape[0] - new_size[0])/2)] = 255
    enlarged_image[:, -int((enlarged_image.shape[0] - new_size[0])/2):-1] = 255

    for i, (polygon_points, polyline_points) in enumerate(zip(polygon_list, polyline_list)):
        if polyline_points is None:
            continue
        polygon_mask = np.zeros_like(enlarged_image)
        cv2.fillPoly(polygon_mask, [polygon_points], (255))
        polygon_occluded, polyline_occluded = recompute_occluded_leaf(enlarged_image, polygon_mask, polyline_points)
        polygon_list[i], polyline_list[i] = polygon_occluded, polyline_occluded

    crop = np.array([int((enlarged_image.shape[0] - new_size[0])/2), int((enlarged_image.shape[1] - new_size[1])/2)])
    polygon_list = [polygon - crop if polygon is not None else polygon for polygon in polygon_list]
    polyline_list = [polyline - crop if polyline is not None else polyline for polyline in polyline_list]
    return polygon_list, polyline_list


def split_to_img_list(split_list):
    img_list = []
    with open(f"{split_list}", "r+") as f:
        img_list = [line.replace("\n", "") for line in f.readlines()]
    return img_list

# ToDo: add some random leaves, not perfectly aligned to root center
# ToDo: Resize plant? Or only sample leaves of specific sizes? Else we will always have big leaves in the plant
# ToDo: Recalculate polylines with exact point spreadings.
def main():
    ov_seed = random.randint(0, 10000)
    print("Overall seed: ", ov_seed)
    random.seed(ov_seed)
    np.random.seed(ov_seed)

    debug = False
    annotation_file = "/home/ronja/data/l515_imgs/RumexLeaves/iNaturalist/annotations_whole_leafs.xml"
    training_split = "/home/ronja/data/l515_imgs/RumexLeaves/iNaturalist/dataset_splits/random_train.txt"
    new_annotation_file = "/home/ronja/data/generated/RumexLeaves/composed_annotations/annotations.xml"

    train_img_files = split_to_img_list(training_split)
    annotations = []
    for train_img_file in train_img_files:
        ann = AnnotationConverter.read_cvat_by_id(annotation_file, train_img_file)
        if ann is not None:
            annotations.append(ann)

    number_of_plants = 100
    for i in range(31, number_of_plants):
        seed = i
        print("Plant seed: ", seed)
        random.seed(seed)
        np.random.seed(seed)

        ref_ann = random.choice(annotations)
        img_width, img_height = int(ref_ann.get_img_width()), int(ref_ann.get_img_height())
        enlarged_image = np.zeros((4000, 4000))
        plant_offset = np.array([random.randint(-int(img_width/4), int(img_width/4)), random.randint(-int(img_height/4), int(img_height/4))])
        plant_pos = [int(enlarged_image.shape[0]/2 + plant_offset[0]), int(enlarged_image.shape[1]/2 + plant_offset[1])]
        polygon_list, polyline_list = compose_plant(annotations, plant_pos)
        polygon_list, polyline_list = sort_leaves_by_distance(polygon_list, polyline_list, plant_pos)

        # image border mask
        polygon_list, polyline_list = handle_occulsions(polygon_list, polyline_list, plant_pos)

        # Crop image size
        polygon_list, polyline_list = crop_image(enlarged_image.shape, (img_height, img_width), polygon_list, polyline_list)

        # Let's generate the corresponding segmentation mask
        labels = {"leaf_blade": 1, "leaf_stem": 2}
        if debug:
            labels = {"leaf_blade": 150, "leaf_stem": 255}
        seg_mask = np.zeros((img_width, img_height), dtype=np.uint8)
        for j, polygon_points in enumerate(polygon_list):
            if polyline_list[j] is None:
                continue
            cv2.fillPoly(seg_mask, [polygon_points], (labels["leaf_blade"]))
            if debug:
                cv2.polylines(seg_mask, [polygon_points], False, (50), 10)
        for polyline_points in polyline_list:
            if polyline_points is not None:
                cv2.polylines(seg_mask, [polyline_points], False, (labels["leaf_stem"]), 25)
                if debug:
                    for point in polyline_points:
                        cv2.circle(seg_mask, (point[0], point[1]), 10, (150), -1)
        cv2.imwrite(f"{os.path.dirname(new_annotation_file)}/{ov_seed}_{seed}.png", seg_mask)

        polygon_annotations = []
        polyline_annotations = []



        for polygon_array, polyline_array in zip(polygon_list, polyline_list):
            if polyline_array is None:
                continue
            pol = Polygon("leaf_blade")
            pol.set_polygon_points_as_array(polygon_array.squeeze())
            polygon_annotations.append(pol)
            pol = Polyline("leaf_stem")
            pol.set_polyline_points_as_array(polyline_array.squeeze())
            polyline_annotations.append(pol)
        
        annotation = Annotation(f"{ov_seed}_{seed}.jpg", img_width, img_height, bb_list=[], polygon_list=polygon_annotations, ellipse_list=[], polyline_list=polyline_annotations)
        AnnotationConverter.extend_cvat(annotation, new_annotation_file)



if __name__ == "__main__":
    main()




