"""
Provides utility functions for generating new data with 
Stable Diffusion image to image generation and inpainting
"""
import sys; sys.path.append("./")

import json
import numpy as np
import cv2
import base64
import requests
import matplotlib.pyplot as plt
import random

from PIL import Image
from scipy.ndimage import binary_dilation

from kalie.utils.utils import descale_point, print_debug, scale_point
from kalie.utils.segmentation import get_scene_object_bboxes, get_segmentation_masks

TOOL = 'tool'
TARGET = 'target'

# from controlnet modules.masking
def expand_crop_region(crop_region, processing_width, processing_height, image_width, image_height):
    """expands crop region get_crop_region() to match the ratio of the image the region will processed in; returns expanded region
    for example, if user drew mask in a 128x32 region, and the dimensions for processing are 512x512, the region will be expanded to 128x128."""

    x1, y1, x2, y2 = crop_region

    ratio_crop_region = (x2 - x1) / (y2 - y1)
    ratio_processing = processing_width / processing_height

    if ratio_crop_region > ratio_processing:
        desired_height = (x2 - x1) / ratio_processing
        desired_height_diff = int(desired_height - (y2-y1))
        y1 -= desired_height_diff//2
        y2 += desired_height_diff - desired_height_diff//2
        if y2 >= image_height:
            diff = y2 - image_height
            y2 -= diff
            y1 -= diff
        if y1 < 0:
            y2 -= y1
            y1 -= y1
        if y2 >= image_height:
            y2 = image_height
    else:
        desired_width = (y2 - y1) * ratio_processing
        desired_width_diff = int(desired_width - (x2-x1))
        x1 -= desired_width_diff//2
        x2 += desired_width_diff - desired_width_diff//2
        if x2 >= image_width:
            diff = x2 - image_width
            x2 -= diff
            x1 -= diff
        if x1 < 0:
            x2 -= x1
            x1 -= x1
        if x2 >= image_width:
            x2 = image_width

    return x1, y1, x2, y2

def read_json(file_path):
    with open(file_path, 'r') as file:
        return json.load(file)
    
def encode_image(img_path: str) -> str:
    img = cv2.imread(img_path)
    _, bytes = cv2.imencode(".png", img)
    encoded_image = base64.b64encode(bytes).decode("utf-8")
    return encoded_image

def encode_bool_array(bool_arr: np.ndarray) -> str:
    # Ensure the input is a boolean NumPy array
    if bool_arr.dtype != np.bool_:
        raise ValueError("Input must be a boolean numpy array")

    uint8_arr = bool_arr.astype(np.uint8) * 255
    _, bytes = cv2.imencode(".png", uint8_arr)
    encoded = base64.b64encode(bytes).decode("utf-8")

    return encoded
   
def generate_payload_from_jsons(filenames):
    # Load all JSON files into separate dictionaries
    data_parts = [read_json(file) for file in filenames]
    
    # Merge all dictionaries into one
    # The later file's data will overwrite earlier ones in case of key conflicts
    # (though there shouldn't really be any)
    full_payload = {}
    for part in data_parts:
        full_payload.update(part)
        
    return full_payload

def show_anns(anns):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)

    img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
    img[:,:,3] = 0
    for ann in sorted_anns:
        m = ann['segmentation']
        color_mask = np.concatenate([np.random.random(3), [0.35]])
        img[m] = color_mask
    ax.imshow(img)
    
def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)
    
def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)   
    
def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2)) 

def expand_mask(mask, n):
    """
    Expands the True areas in a boolean mask by 'n' pixels.

    Parameters:
        mask (np.ndarray): A 2D NumPy array of boolean values, where True indicates the masked area.
        n (int): The number of pixels by which to expand the mask.

    Returns:
        np.ndarray: A new mask array with the True areas expanded by 'n' pixels.
    """
    # Create a structuring element that defines the "connectivity" or shape of the expansion
    # For example, a square structuring element with size (2*n+1, 2*n+1)
    structuring_element = np.ones((2*n+1, 2*n+1), dtype=bool)
    
    # Apply binary dilation
    expanded_mask = binary_dilation(mask, structure=structuring_element)
    
    return expanded_mask

def get_points_in_crop_region(points, crop_region, img_size):
    """
    Given a dictionary of points and a crop region, returns the points that are within the crop region
    """
    x1, y1, x2, y2 = crop_region
    pts_in_crop = {}
    for k, v in points.items():
        v_descaled = descale_point(v, img_size[0], img_size[1])
        if x1 <= v_descaled[0] <= x2 and y1 <= v_descaled[1] <= y2:
            pts_in_crop[k] = v
    return pts_in_crop

def get_points_in_segmask(points, segmask, img_size):
    """
    Given a dictionary of points and a segmentation mask, returns the points that are within the mask
    """
    # First convert PIL image segmask to numpy array
    segmask = np.array(segmask)
    h, w = segmask.shape
    pts_in_mask = {}
    for k, v in points.items():
        v_descaled = descale_point(v, img_size[0], img_size[1])
        x, y = v_descaled
        x = int(x * w / img_size[0])
        y = int(y * h / img_size[1])
        if segmask[y, x]:
            pts_in_mask[k] = v
    if "Target Point" in pts_in_mask:
        # add "Pre-contact Point" and "Post-contact Point" to pts_in_mask
        pts_in_mask["Pre-contact Point"] = points["Pre-contact Point"]
        pts_in_mask["Post-contact Point"] = points["Post-contact Point"]
        
    return pts_in_mask

def get_segmasks(obs_image, all_object_names, sam_checkpoint="ckpts/sam_vit_h_4b8939.pth", logdir="intermediate_results/"):
        
    boxes, logits, phrases = get_scene_object_bboxes(
        obs_image, 
        all_object_names,
        visualize=True,
        logdir=logdir)

    segmasks = get_segmentation_masks(
        obs_image, 
        all_object_names, 
        boxes, 
        logits, 
        phrases,
        sam_checkpoint,
        visualize=True,
        logdir=logdir)
    
    return segmasks
    

def call_api(url: str, payload: dict, outimgpath):
    """
    Given an image2image payload, generates an image with the given
    parameters
    """
    response = requests.post(url=url, json=payload).json()
    if "images" not in response:
        print(response)
    else:
        with open(outimgpath, 'wb') as f:
            print(f"writing to {outimgpath}")
            f.write(base64.b64decode(response['images'][0]))

def get_mask_padding(img2img_params_file):
    return json.load(open(img2img_params_file))['inpaint_full_res_padding']
    
def generate_random_transformation(segmask, img2img_params_file, logdir, debug=False):
    """
    Generates a random transformation to apply to the image.
    """
    if debug:
        print("GENERATING IN DEBUG MODE")
        # read from file 
        with open(f'{logdir}debug_params.json', 'r') as file:
            transform_params = json.load(file)
            scale_factor_x = transform_params['scale_factor_x']
            scale_factor_y = transform_params['scale_factor_y']
            angle = transform_params['angle']
            translate_x = transform_params['translate_x']
            translate_y = transform_params['translate_y']
    else:
        scale_factor_x = random.uniform(0.8, 1.2)
        scale_factor_y = random.uniform(0.8, 1.2)
        angle = random.uniform(-20, 20)
        translate_x = random.uniform(-20, 20) # this is in the original pixel space (i.e. 700 x 700)
        translate_y = random.uniform(-20, 20)
    
    padding = get_mask_padding(img2img_params_file)
    with open(img2img_params_file, 'r') as file:
        height = json.load(file)['height']
        
    expand_mask(segmask, 10)
    segmask = segmask if isinstance(segmask, Image.Image) else Image.fromarray(segmask)
    
    if box := segmask.getbbox():
        x1, y1, x2, y2 = box
        crop_region = (max(x1 - padding, 0), max(y1 - padding, 0), min(x2 + padding, segmask.size[0]), min(y2 + padding, segmask.size[1])) if padding else box
    
    # print(crop_region)
    crop_height = crop_region[3] - crop_region[1]
    crop_width = crop_region[2] - crop_region[0]
    
    scale_factor = 512 / (2 * padding + max(crop_height, crop_width))
    # scale_factor = height / max(crop_height, crop_width)
    new_translate_x = translate_x * scale_factor
    new_translate_y = translate_y * scale_factor
    # print_green(f"Scale factor: {scale_factor}, Original translate: ({translate_x}, {translate_y}), New translate: ({new_translate_x}, {new_translate_y})")
    
    # write these to the file
    transform_params = {
        "scale_factor_x": scale_factor_x,
        "scale_factor_y": scale_factor_y,
        "angle": angle,
        "translate_x": int(new_translate_x),
        "translate_y": int(new_translate_y)
    }
    
    with open(f'{logdir}transform_params.json', 'w') as file:
        json.dump(transform_params, file, indent=4)
        
    return scale_factor_x, scale_factor_y, angle, translate_x, translate_y, crop_region


def generate_specific_transformation(segmask, img2img_params_file, logdir, scale_factor_x, scale_factor_y, angle, translate_x, translate_y):
    padding = get_mask_padding(img2img_params_file)
        
    expand_mask(segmask, 10)
    segmask = segmask if isinstance(segmask, Image.Image) else Image.fromarray(segmask)
    
    if box := segmask.getbbox():
        x1, y1, x2, y2 = box
        crop_region = (max(x1 - padding, 0), max(y1 - padding, 0), min(x2 + padding, segmask.size[0]), min(y2 + padding, segmask.size[1])) if padding else box
    
    crop_height = crop_region[3] - crop_region[1]
    crop_width = crop_region[2] - crop_region[0]
    
    scale_factor = 512 / (2 * padding + max(crop_height, crop_width))
    new_translate_x = translate_x * scale_factor
    new_translate_y = translate_y * scale_factor
    
    transform_params = {
        "scale_factor_x": scale_factor_x,
        "scale_factor_y": scale_factor_y,
        "angle": angle,
        "translate_x": int(new_translate_x),
        "translate_y": int(new_translate_y)
    }
    
    with open(f'{logdir}transform_params.json', 'w') as file:
        json.dump(transform_params, file, indent=4)
        
    return scale_factor_x, scale_factor_y, angle, translate_x, translate_y, crop_region

def generate_transformation_array(minval, maxval, num_samples):
    # generate smooth values between minval and maxval, starting from the middle and going towards min_val, then going back through the middle to max_val, then back to the middle
    avg_val = (minval + maxval) / 2
    val_range = maxval - minval
    samples = []
    for i in range(num_samples):
        if i < num_samples // 4:
            samples.append(avg_val - val_range * (i / (num_samples // 2)))
        elif i < num_samples // 2:
            samples.append(avg_val + val_range * ((i - num_samples // 2) / (num_samples // 2)))
        elif i < 3 * num_samples // 4:
            samples.append(avg_val + val_range * ((i - num_samples // 2) / (num_samples // 2)))
        else:
            samples.append(avg_val - val_range * ((i - num_samples) / (num_samples // 2)))
    return samples

def update_keypoints(points, crop_region, segmask, scale_x, scale_y, angle, translate_x, translate_y, img_size, debug=False):
    
    points_in_segmask = get_points_in_segmask(points, segmask, img_size)
# print(f"Points in Segmask: {points_in_segmask}")
    
    # scale crop region
    x1, y1, x2, y2 = crop_region
    p1 = scale_point((x1, y1), img_size[0], img_size[1])
    p2 = scale_point((x2, y2), img_size[0], img_size[1])
    center = ((p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2)
    
    print_debug(f"Center: {center}, Descaled: {descale_point(center, img_size[0], img_size[1])}", debug)
    
    angle = -np.deg2rad(angle)
    
    for pt_name, pt_val in points_in_segmask.items():
        x, y = pt_val
        
        print_debug(f"orig val: ({x}, {y})", debug)
        
        x -= (center[0])
        y -= (center[1])
        
        print_debug(f"centered val: ({x}, {y})", debug)
        
        x = int(x * scale_x)
        y = int(y * scale_y)
        
        print_debug(f"scaled val: ({x}, {y})", debug)
        temp_x = x
        x = int(x * np.cos(angle) - y * np.sin(angle))
        y = int(temp_x * np.sin(angle) + y * np.cos(angle))
        
        print_debug(f"rotated val: ({x}, {y})", debug)
        
        translate_x_scaled, translate_y_scaled = scale_point((translate_x, translate_y), img_size[0], img_size[1])
        x += translate_x_scaled
        y += translate_y_scaled
        print_debug(f"translated val: ({x}, {y})", debug)
        
        x += (center[0])
        y += (center[1])
        
        print_debug(f"final val: ({x}, {y})", debug)
        points[pt_name] = [int(x), int(y)]
        print("***********")
    
    return points

def generate_image(payload, out_img_path, url):
    """
    Generates and returns a new image generated by SD based on the
    original image, prompt, and mask (for inpainting)
    img_path: path to initial image
    """
    call_api(url, payload, out_img_path)
    
def generate_payload(img2img_params_filename, prompt_params_filename, in_img_path, prompt, segmask=None, controlnet_filename=None, load_contour=False):
    full_payload = {}
    full_payload.update(read_json(img2img_params_filename))
    full_payload.update(read_json(prompt_params_filename))
    
    input_image = encode_image(in_img_path)
    
    full_payload["prompt"] = prompt
    full_payload["init_images"] = [input_image]
    
    if controlnet_filename:
        full_payload["alwayson_scripts"] = read_json(controlnet_filename)
        full_payload["load_contour"] = load_contour
        full_payload["use_transformation"] = True
        
    mask_image = encode_bool_array(expand_mask(segmask, 10))
        
    full_payload["mask"] = mask_image
            
    return full_payload

def update_new_data(new_data_json, new_key, new_img_path, new_points):
    """
    Helper function to update new_data_json with new data.
    """
    
    new_data_json[new_key] = {
        "img": new_img_path,
        "points": new_points
    }
