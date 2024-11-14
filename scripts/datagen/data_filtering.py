"""
This is an optional file that can be used to filter out invalid generations from a dataset.

This script queries GPT-4o and asks whether the given image could be used to complete the task instruction. If the response is "Yes", the image is kept in the dataset. Otherwise, it is removed.
"""

import sys; sys.path.append("./")
from kalie.utils.gpt_utils import request_gpt
from kalie.utils.utils import convert_json_to_str_format
import os
import json
import time

from shutil import copy2
from absl import app, flags

FLAGS = flags.FLAGS

flags.DEFINE_string('input_archive_path', './archive_2024-08-06_sweeping_canny/', 'Archive from which to pull original data')
flags.DEFINE_string('output_archive_path', './archive_2024-08-06_sweeping_canny_filtered/', 'Archive into which we dump new data')

flags.DEFINE_string('objs', '[brush, trash]', 'Objects to identify in the task instruction')

flags.DEFINE_string('model', 'gpt-4o-2024-05-13', 'Model to use for GPT requests')
flags.DEFINE_string('sys_context', 'You are a helpful assistant. You will receive an image, taken from a top-down robot\'s view over a table. You will respond to all questions concisely and in the format requested.', 'System (meta) context for GPT requests')

YES = "Yes"
NO = "No"

def check_valid(image_path, task_instruction, objects_to_identify):
    prompt = f"Could this image reasonably correspond to the task instruction \"{task_instruction}\"? Make sure you can identify all objects in the list {objects_to_identify}, one of each, and make sure none of these objects are blurry, extremely malformed, or duplicated."
    
    print(prompt)

    return request_gpt(prompt, [image_path], FLAGS.sys_context, FLAGS.model, local_image=True)

def shorten_response(response_str):
    """
    Take a longer gpt response and shorten it to just "Yes" or "No"
    """
    return YES if YES in response_str else NO

def get_objects_in_task(task_instruction):
    """
    Given a task instruction, query gpt to extract the objects that need to be identified
    """
    prompt = f"You will receive a robot task instruction. List objects that would appear in the scene based on the instruction. Your response should be a list of strings, where each string is the name of an object. Output format: [<object1>, <object2>, ...]. Task instruction: \"{task_instruction}\". "
    system_context = "You are a helpful assistant."
    return request_gpt(prompt, [], system_context, FLAGS.model, local_image=False)


def main(_):
    start_time = time.time()
    
    input_json_filepath = FLAGS.input_archive_path + "data_jsonformat.json"
    output_json_filepath = FLAGS.output_archive_path + "data_jsonformat.json"
    output_str_filepath = FLAGS.output_archive_path + "data_strformat.json"
    os.makedirs(f"{FLAGS.output_archive_path}imgs/", exist_ok=True)
    
    # Load data info from json file
    with open(input_json_filepath, 'r') as file:
        data = json.load(file)
        
    # For each entry, check if it's valid. If it is, add to new data
    new_data = {}
    new_data_str = {}
    for key, value in data.items():
        
        curr_time = time.time()
        
        img = value["img"]
        other = value["other"]
        task_instruction = other["task"]
        objs = FLAGS.objs
        response = check_valid(img, task_instruction, objs)
        
        if shorten_response(response) == YES:
            print(f"Valid entry: {key}")
            new_value = value.copy()
            new_img_path = FLAGS.output_archive_path + f"imgs/{key}.png"
            new_value["img"] = new_img_path
            new_data[key] = new_value
            
            new_data_str[key] = {
                "img": new_img_path,
                "points": convert_json_to_str_format(value['points']),
                "other": value['other']
            }
            
            # Copy over image into new archive
            copy2(img, new_img_path)
        else:
            print(f"Invalid entry: {key}")
        
        print(f"Time taken for entry {key}: {time.time() - curr_time} seconds")    
        print("*************************")
         # Save new data to json file
        with open(output_json_filepath, 'w') as file:
            json.dump(new_data, file, indent=4)
            
        with open(output_str_filepath, 'w') as file:
            json.dump(new_data_str, file, indent=4)
    
    # Save new data to json file
    with open(output_json_filepath, 'w') as file:
        json.dump(new_data, file, indent=4)
        
    with open(output_str_filepath, 'w') as file:
        json.dump(new_data_str, file, indent=4)
        
    print(f"Total time taken: {time.time() - start_time} seconds")

if __name__ == '__main__':
    app.run(main)