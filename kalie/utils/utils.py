import re 
import numpy as np
import json
from torchvision import tv_tensors
import torch
from torchvision.transforms import v2
from PIL import Image
import random
import os

"""
Stores prediction and label strings into json file
"""
def store_pred_label_strs(dir, filepath, pred_label_str):
    os.makedirs(dir, exist_ok=True)

    try:
        with open(os.path.join(dir, filepath), "r") as f:
            existing_data = json.load(f)
    except FileNotFoundError:
        existing_data = []
 
    if pred_label_str not in existing_data:
        existing_data.append(pred_label_str)

    with open(os.path.join(dir, filepath), "w") as f:
        json.dump(existing_data, f, indent=4)
        

# get the center of a bounding box - transform bounding boxes back to the respective 5 points 
def bounding_box_center(bbox):
    x1, y1, x2, y2 = bbox
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    return cx, cy

# scale to be in the range 000 to 999 as per CogVLM training (negative values ok if outside crop)
def scale_point(point, width, height): 
    point_x_scaled = round(point[0] * 1000 / width)
    point_y_scaled = round(point[1] * 1000 / height)
    return [point_x_scaled, point_y_scaled]

# descale to convert coordinates relative to the image
def descale_point(point, width, height): 
    point_x_scaled = round(point[0] * width / 1000)
    point_y_scaled = round(point[1] * height / 1000)
    return [point_x_scaled, point_y_scaled]

"""
Gets prediction and label dictionaries, where pred_label_str is 
EITHER a prediction or a label of the format
Grasp Point: (397, 293),  Function Point: (719, 435),  Target Point: (645, 414),  Pre-contact Point: (535, 425),  Post-contact Point: (833, 464)

returns points_dict of form
{
    "Grasp Point": (x, y),
    "Function Point": (x, y),
    "Target Point": (x, y),
    "Pre-Contact Point": (x, y),
    "Post-Contact Point": (x, y)
}
"""
def str_to_points_dicts(pred_or_label_str):
    points_dict = {}

    # Regex to find points
    point_pattern = re.compile(r"([A-Za-z-]+ Point): \((\d+),\s*(\d+)\)")

    # Parsing points
    for match in re.finditer(point_pattern, pred_or_label_str):
        points_dict[match.group(1)] = (int(match.group(2)), int(match.group(3)))

    return points_dict


def array_to_points_dicts(pred_or_label_str):
    points_dict = {
    "Grasp Point": (pred_or_label_str[0], pred_or_label_str[1]),
    "Function Point": (pred_or_label_str[2], pred_or_label_str[3]),
    "Target Point": (pred_or_label_str[4], pred_or_label_str[5]),
    "Pre-contact Point": (pred_or_label_str[6], pred_or_label_str[7]),
    "Post-contact Point": (pred_or_label_str[8], pred_or_label_str[9])
    }

    return points_dict

"""
Gets prediction / label dictionary from JSON formatted string

returns either pred_points_dict or label_points_dict of form
{
    "Grasp Point": [x, y],
    "Function Point": [x, y],
    "Target Point": [x, y],
    "Pre-Contact Point": [x, y],
    "Post-Contact Point": [x, y]
}
"""
def json_to_points_dicts(pred_or_label_json_str):
    try: 
        points_dict = json.loads(pred_or_label_json_str)
        return points_dict
    except: 
        return {} 
    
def calculate_errors(pred, label, formatstyle="str"):
        # Parse the prediction and label points
        if formatstyle == "str":
            pred_points, label_points = str_to_points_dicts(pred), str_to_points_dicts(label)
        elif formatstyle == "array":
            pred_points, label_points = array_to_points_dicts(pred), array_to_points_dicts(label)
        else:
            pred_points, label_points = json_to_points_dicts(pred), json_to_points_dicts(label)

        mse_values = {}
        abs_error_xs = {}
        abs_error_ys = {}
        total_mse = 0
        if len(pred_points) == 0:
            return {"Grasp Point": 1000000, "Function Point": 1000000, "Target Point": 1000000, "Pre-contact Point": 1000000, "Post-contact Point": 1000000}, 1000000, {"Grasp Point": 500, "Function Point": 500, "Target Point": 500, "Pre-contact Point": 500, "Post-contact Point": 500}, {"Grasp Point": 500, "Function Point": 500, "Target Point": 500, "Pre-contact Point": 500, "Post-contact Point": 500}# temp

        # Calculate errors for each point
        for point_type in pred_points:
            pred_x, pred_y = pred_points[point_type]
            label_x, label_y = label_points[point_type]

            pred_x, pred_y = int(float(pred_x)), int(float(pred_y))
            label_x, label_y = int(float(label_x)), int(float(label_y)) 
            
            mse = ((pred_x - label_x) ** 2 + (pred_y - label_y) ** 2) / 2
            abs_err_x = abs(pred_x - label_x)
            abs_err_y = abs(pred_y - label_y)
            
            mse_values[point_type] = mse
            abs_error_xs[point_type] = abs_err_x
            abs_error_ys[point_type] = abs_err_y
            total_mse += mse

        # Calculate average MSE
        average_mse = total_mse / len(pred_points)

        return mse_values, average_mse, abs_error_xs, abs_error_ys

def convert_json_to_str_format(input_string):
    """
    Convert a JSON-like string of keypoints into a formatted string.

    Parameters:
        input_string (str): A string in JSON format containing keypoints.
            (ex. "{
                    "Grasp Point": [x, y],
                    "Function Point": [x, y],
                    "Target Point": [x, y],
                    "Pre-Contact Point": [x, y],
                    "Post-Contact Point": [x, y]
                }")

    Returns:
        str: A formatted string representing the keypoints. 
            (ex. "Grasp Point: (176, 634), Function Point: (359, 412), Target Point: (484, 585), Pre-contact Point: (443, 495), Post-contact Point: (477, 793)")
    """
    # Parse the JSON-like string into a Python dictionary
    keypoint_dict = json.loads(input_string)

    # Build the output string using a list comprehension and the `join` method
    formatted_string = ', '.join(f"{key}: ({value[0]}, {value[1]})" for key, value in keypoint_dict.items())
    
    return formatted_string

# runs the full image augmentation process given an original image + five points 
def get_image_transformation(orig_image, orig_points, w, h, min_size_ratio): 
    transformations = get_transformations(orig_image.size[0], orig_image.size[1], min_size_ratio) 
    transformed_image, transformed_points, transformed_center = transform_image_and_points(transformations, orig_image, orig_points)
    array_img = np.asarray(transformed_image)
    cropped_image, crop = deterministic_crop(array_img, (w, h), transformed_center)
    cropped_points = [transform(point, crop) for point in transformed_points]
    return cropped_image, cropped_points


def crop_evaluation_data(orig_image, orig_points, w, h): 
    width, height = orig_image.size
    center_x = width / 2
    center_y = height / 2
    array_img = np.asarray(orig_image)
    cropped_image, crop = deterministic_crop(array_img, (w, h), (center_x, center_y))
    cropped_points = [transform(point, crop) for point in orig_points]
    return cropped_image, cropped_points

def crop_realtime_eval(orig_image, w, h): 
    width, height = orig_image.size
    center_x = width / 2
    center_y = height / 2
    array_img = np.asarray(orig_image)
    cropped_image, crop = deterministic_crop(array_img, (w, h), (center_x, center_y))
    return cropped_image, crop

# transform the 5 points onto the original image  
def transform_original(point, crop):
    point = list(point)
    point[0] += crop[1]
    point[1] += crop[0]
    return point

# get the torch transformations we apply 
def get_transformations(W_orig, H_orig, min_size_ratio): 
    r = random.uniform(min_size_ratio, 1.0)
    transformations = v2.Compose([ 
        v2.RandomRotation(degrees=15),
        v2.RandomResizedCrop(size=(int(r*W_orig), int(r*H_orig)), scale=(0.9,1)),
        v2.RandomHorizontalFlip(p=0.5),
        v2.RandomVerticalFlip(p=0.5), 
        v2.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.2, hue=0.2),
    ])
    return transformations

# apply the given torch transformations on the image, 5 points, and the image center 
def transform_image_and_points(transformations, orig_image, orig_points):
    H, W = orig_image.size[1], orig_image.size[0]
    center = bounding_box_center([0, 0, W, H])
    orig_points.append([int(center[0]), int(center[1])])   # append the center
    boxes = generate_bounding_boxes(orig_points, 5)
    boxes = tv_tensors.BoundingBoxes(boxes, format="XYXY", canvas_size=(H, W))
    output_dict = transformations({"image": orig_image, "boxes": boxes})
    points = []
    for box in output_dict["boxes"]:
        points.append((bounding_box_center(box.tolist())))
    return output_dict["image"], points[:-1], points[-1] 

# deterministic crop applied on the transformed center 
def deterministic_crop(image, output_size, center):
    output_w, output_h = output_size
    crop = [max(0, int(center[0] - output_w / 2)), max(0, int(center[1] - output_h / 2)), min(int(center[0] + output_w / 2), image.shape[0]), min(int(center[1] + output_h / 2), image.shape[1])]
    cropped_image = (image[crop[0]:crop[2], crop[1]:crop[3]].copy())
    return Image.fromarray(cropped_image), crop 

# transform the 5 points onto the deterministic crop 
def transform(point, crop):
    point = list(point)
    point[0] -= crop[1]
    point[1] -= crop[0]
    return point

# torch transformations only accepts bounding boxes, so this function generates a small 
# bounding box for each point
def generate_bounding_boxes(points, box_size): 
    num_points = len(points)
    half_size = box_size / 2
    boxes = torch.zeros(num_points, 4)
    for i in range(num_points):
        x, y = points[i]
        x1 = x - half_size
        y1 = y - half_size
        x2 = x + half_size
        y2 = y + half_size
        boxes[i] = torch.tensor([x1, y1, x2, y2])
    return boxes

def strip_non_alpha(s):
    """
    Removes non-alphabet characters from the start and end of a string.
    Parameters:
        s (str): The string from which to remove non-alphabet characters.

    Returns:
        str: The modified string with non-alphabet characters removed from the start and end.
    """
    # Regular expression to match non-alphabet characters at the start (^) and end ($) of the string
    return re.sub(r'^[^a-zA-Z]+|[^a-zA-Z]+$', '', s)

def unique_ordered(items):
    """
    Returns a list of unique items in the order they first appeared in the input list.

    Parameters:
        items (list of str): The input list of strings.

    Returns:
        list of str: A list containing only the first occurrence of each string in the input list.
    """
    seen = set()
    unique_items = []
    for item in items:
        if item not in seen:
            unique_items.append(item)
            seen.add(item)
    return unique_items

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    
def print_fail(content):
    print(bcolors.FAIL + str(content) + bcolors.ENDC)
    
def print_green(content):
    print(bcolors.OKGREEN + str(content) + bcolors.ENDC)
    
def print_warning(content):
    print(bcolors.WARNING + str(content) + bcolors.ENDC)
    
def print_debug(content, debug, warning=False):
    if warning and debug:
        print_warning(str(content))
    elif debug:
        print(str(content))
