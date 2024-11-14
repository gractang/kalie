"""
Generates new data from existing data by querying GPT-4o and using diffusion to generate new images.

Usage:
First run `./webui.sh --no-half --api` in the stable-diffusion-webui repo
Then run `python scripts/datagen/data_generation.py` from the root directory.

The script will generate new images based on the data in the input archive. It will use an LLM to generate prompts for each object in the image, then use diffusion to generate new images based on those prompts. The new images will be saved in the output archive.
"""
import sys; sys.path.append("./")
import random
import time
import json
import os

from kalie.utils.utils import convert_json_to_str_format, unique_ordered, print_warning
from kalie.utils.data_generation_utils import get_segmasks, generate_image, generate_random_transformation, update_keypoints, generate_payload
from kalie.utils.gpt_utils import request_gpt

from shutil import copy2
from absl import app, flags
from PIL import Image
import numpy as np

FLAGS = flags.FLAGS

flags.DEFINE_string('inpainting_params_path', 'scripts/datagen/configs/inpainting_img2img_params.json', 'Configuration for inpainting')
flags.DEFINE_string('prompt_params_path', 'scripts/datagen/configs/prompt_params.json', 'Configuration for prompts')
flags.DEFINE_string('ctrlnet_conf', 'scripts/datagen/configs/ctrlnet_params_softedge.json', 'Controlnet configuration file')

flags.DEFINE_integer('num_prompts', 1, 'Number of prompts to generate per image')
flags.DEFINE_integer('num_gpt_prompts', 5, 'Number of prompts to generate with GPT')
flags.DEFINE_integer('max_attempts', 3, 'Max number of times to try querying GPT before giving up')
flags.DEFINE_float('gpt_temp', 1.3, 'Temperature to use for GPT-4o when generating prompts')

flags.DEFINE_string('seg_model_path', 'models/sam_vit_h_4b8939.pth', 'Segmentation model path')
flags.DEFINE_string('url', 'http://localhost:7860/sdapi/v1/img2img', 'Endpoint for API')
flags.DEFINE_string('gpt_model', 'gpt-4o-2024-05-13', 'GPT model to use for generating prompts')

flags.DEFINE_string('sys_context', 'You are a helpful assistant. You will receive an image, taken from a top-down robot\'s view, as well as a task instruction. You will respond to all questions concisely and in exactly the format requested.', 'System context to use for GPT')
flags.DEFINE_string('addon_prompt', ', top down view, highres, best quality, photorealistic', 'Additional prompt for diffusion to append to end of generated prompt')

flags.DEFINE_string('input_archive_path', './input_archive/', 'Archive from which to pull original data')
flags.DEFINE_string('output_archive_path', './output_archive/', 'Archive into which we dump new data')

flags.DEFINE_string('logdir', 'intermediate_results/', 'Directory to store intermediate results')

flags.DEFINE_boolean('debug', False, 'Debugging mode.')
flags.DEFINE_boolean('gpt_get_prompts', True, 'Whether to use GPT to get prompts.')

DEVICE = 'cuda'
MODEL_TYPE = 'vit_h'
TEMP_DIR = 'intermediate_results/'
TEMP_PATH = f'{TEMP_DIR}temp.png'

def print_debug(content, warning=False):
    if FLAGS.debug:
        if warning:
            print_warning(content)
        else:
            print(content)
    
def remove_obj_from_obj_list(object_name, object_list):
    """
    Removes the object_name string from a list of objects
    """
    return [obj for obj in object_list if obj != object_name]

def query_gpt_for_objects(task_instruction, img_path):
    """
    Given a task instruction, query gpt to extract relevant and distractor objects
    Currently replaced by manually defined objects.
    
    Parameters:
        task_instruction (str): The task instruction for the image
        img_path (str): The path to the image to use for the query
        
    Returns:
        Dict[str, List[str]]: A dictionary with two keys: "relevant" and "distractor". The values
        are lists of strings, where each string is the name of an object.
    """
    prompt = f"You will receive a robot task instruction, as well as an image, taken from a top-down robot's view. Describe objects that appear in the image that would be relevant to the task instruction. Be general in your response. Your response should be a list of strings, where each string is the name of an object. Output format: [\"<object1>\", \"<object2>\", ... etc.]. Task instruction: \"{task_instruction}\"."
    system_context = "You are a helpful assistant."
    response = request_gpt(prompt, [img_path], system_context, FLAGS.gpt_model, local_image=True)
    relevant_objs = remove_obj_from_obj_list("table", json.loads(response))
    
    prompt = f"You will receive a robot task instruction, as well as an image, taken from a top-down robot's view. Describe objects that appear on the table in the image that are NOT used in the task instruction and should NOT be interacted with. Make sure these objects cannot fall under any of the categories of objects that exist in the task instruction. Your response should be a list of strings, where each string is the name of an object. If none such objects exist, simply return an empty list \"[]\". Output format: [\"<object1>\", \"<object2>\", ... etc.]. Task instruction: \"{task_instruction}\"."
    response = request_gpt(prompt, [img_path], system_context, FLAGS.gpt_model, local_image=True)
    distractor_objs = remove_obj_from_obj_list("table", json.loads(response))
    
    return {"relevant": relevant_objs, "distractor": distractor_objs}

def query_gpt_for_prompts(image, task_instruction, obj_name, model, n, temperature):
    """
    Queries an LLM to generate n prompts to use to generate more data with diffusion.
    
    Parameters:
        model (str): The name of the LLM model to query (ex. "gpt-4o-2024-05-13").
        n (int): The number of prompts to generate.
    
    Returns:
        List[str]: A list of n prompts generated by the LLM.
    """
    prompt = f"This image, taken from a robot\'s point of view, corresponds to the task \"{task_instruction}\" For the object {obj_name} in the image, consider its object category. Then, sample {n} reasonable descriptions of the object with a different appearance but the same category. The description can include (a subset of) details such as colors, textures, and materials. Your final output should be a list of strings of object variant descriptions. Output format: \"[\"<description 1>\", \"<description 2>\", ... etc.]\"."
    attempts = 0
    while attempts < FLAGS.max_attempts:
        attempts += 1
        try:
            response = request_gpt(prompt, image, FLAGS.sys_context, model, True, temperature)
            print("*** Response from GPT: ***")
            print(response)
            # Attempt to load the string as JSON
            result = json.loads(response)
            
            # Check if the result is a list
            if isinstance(result, list):
                return result
            else:
                print(f"Received valid JSON but not a list: {response}")
        except json.JSONDecodeError:
            print(f"Received invalid JSON: {response}")
        
        print("Retrying...")
    
    raise Exception("Failed to retrieve a valid JSON list after several attempts.")
    
    return json.loads(response)
    
def get_diffusion_prompts(image_path, task_instruction, obj_name, n, num_prompts, model, temperature=1):
    """
    Generates the final list of num_prompts prompts to use for data generation.
    Queries LLM to get n prompts, then randomly selects k=num_prompts prompts from the list
    (without replacement) to increase diversity in prompts.
    
    Parameters:
        image_path (str): The path to the image to use for the prompts.
        task_instruction (str): The task instruction for the prompts.
        obj_name: The object for which to generate prompts (ex. "brush")
        n (int): The number of prompts for the LLM to generate.
        num_prompts (int): The number of prompts to return. Typically, num_prompts < n.
        temperature (float): The temperature to use for the LLM.
        model (str): The name of the LLM model to query.
    
    Returns:
        List[str]: A list of num_prompts prompts to use for data generation.
    """
    all_n_prompts = query_gpt_for_prompts(image_path, task_instruction, obj_name, model, n, temperature)
    
    if len(all_n_prompts) < num_prompts:
        return all_n_prompts
    return random.sample(all_n_prompts, k=num_prompts)

def update_new_data(new_data_json, new_key, new_img_path, new_points, new_other):
    """
    Helper function to update new_data_json with new data.
    """
    new_data_json[new_key] = {
        "img": new_img_path,
        "points": new_points,
        "other": new_other
    }
    
def main(_):
    start_time = time.time()
    curr_start_time = time.time()
    gen_img_time = 0
    
    json_path = f"{FLAGS.input_archive_path}data_jsonformat.json"
    os.makedirs(f"{FLAGS.output_archive_path}imgs/", exist_ok=True)
    os.makedirs(f"{FLAGS.logdir}", exist_ok=True)
    
    with open(json_path, 'r') as file:
        data = json.load(file)
        
    new_data_json = {}
    
    for entry, val in data.items():
        
        # first copy over original data to new data
        orig_img_path = val['img']
        new_key = f"entry_{len(new_data_json)}"
        new_img_path = f"{FLAGS.output_archive_path}imgs/entry_{len(new_data_json)}.png"
        
        update_new_data(new_data_json, new_key, new_img_path, val['points'], val['other'])
        
        copy2(orig_img_path, new_img_path)
        
        # if val['other']['objects'] exists, load it into objects_in_task
        if 'objects' in val['other']:
            objects_in_task = val['other']['objects']
        else:
            print("No way to get objects")
            exit()
        
        all_prompts = {}
        
        segmask_start_time = time.time()
        init_image = Image.open(orig_img_path)
        all_segmasks = get_segmasks(init_image, objects_in_task, sam_checkpoint=FLAGS.seg_model_path, logdir=FLAGS.logdir)
        
        print(f"Getting segmentation masks for original entry {entry} took {time.time() - segmask_start_time} seconds")
        
        if FLAGS.gpt_get_prompts:
            for obj in objects_in_task:
                task_instruction = val['other']['task']
                
                # Get prompts
                prompts = get_diffusion_prompts(new_img_path, task_instruction, obj, FLAGS.num_gpt_prompts, FLAGS.num_prompts, FLAGS.gpt_model, FLAGS.gpt_temp)
                all_prompts[obj] = prompts
                
            # Save all_prompts to old data
            val['other']['prompts'] = all_prompts
            
        # If val['other']['prompts'] exists, load it into all_prompts
        elif 'prompts' in val['other']:
            all_prompts = val['other']['prompts']
        else:
            print("No way to get prompts")
            exit()
                
        print("************************")
        print("Task: ", val['other']['task'])
        print("Objects: ", objects_in_task)
        print("Prompts: ", all_prompts)
        print("************************")
        
        for i in range(FLAGS.num_prompts):
            curr_start_time = time.time()
            
            new_key = f"entry_{len(new_data_json)}"
            new_img_path = f"{FLAGS.output_archive_path}imgs/entry_{len(new_data_json)}.png"
            prompts = []
            new_keypoints = val["points"]
            
            for j, obj in enumerate(objects_in_task):
                obj_start_time = time.time()
                prompts.append(all_prompts[obj][i])
                diffusion_prompt = all_prompts[obj][i] + FLAGS.addon_prompt
                print(f"Generating {new_key} object \"{obj}\" with diffusion prompt \"{diffusion_prompt}\"")
                
                # Depending on which stage of the generation process we are in, we want to save to different
                # paths. The gist is that we sequentially generate objects, so we want the first generation
                # to take in the original image as input and generate a new image, then the second generation
                # should take in the first generation as input, and so on. Then the final image should take in
                # the penultimate image as input and save it to the final output image.
                segmask = all_segmasks[obj]["mask"]
                
                scale_x, scale_y, angle, translate_x, translate_y, crop_region = generate_random_transformation(segmask, FLAGS.inpainting_params_path, FLAGS.logdir)
                
                # update keypoints in crop_region based on transformations
                points_dict = json.loads(new_keypoints)
                new_points_dict = update_keypoints(points_dict, crop_region, segmask, scale_x, scale_y, angle, translate_x, translate_y, init_image.size)
                points_format_js = json.dumps(new_points_dict, indent=4)
                new_keypoints = f"""{points_format_js}"""
                
                if j == 0 and j == len(objects_in_task) - 1:
                    # print("********* Only *********")
                    payload = generate_payload(FLAGS.inpainting_params_path, FLAGS.prompt_params_path, orig_img_path, diffusion_prompt, segmask, FLAGS.ctrlnet_conf, load_contour=False)
                    generate_image(payload, new_img_path, FLAGS.url)
                else:
                    if j == 0:
                        # print("********* First *********")
                        payload = generate_payload(FLAGS.inpainting_params_path, FLAGS.prompt_params_path, orig_img_path, diffusion_prompt, segmask, FLAGS.ctrlnet_conf, load_contour=False)
                        generate_image(payload, TEMP_PATH, FLAGS.url)
                    elif j == len(objects_in_task) - 1:
                        # print("********* Last *********")
                        payload = generate_payload(FLAGS.inpainting_params_path, FLAGS.prompt_params_path, TEMP_PATH, diffusion_prompt, segmask, FLAGS.ctrlnet_conf, load_contour=True)
                        generate_image(payload, new_img_path, FLAGS.url)
                    else:
                        # print("********* Middle *********")
                        payload = generate_payload(FLAGS.inpainting_params_path, FLAGS.prompt_params_path, TEMP_PATH, diffusion_prompt, segmask, FLAGS.ctrlnet_conf, load_contour=True)
                        generate_image(payload, TEMP_PATH, FLAGS.url)
                    
                print(f"Generating {new_key} object \"{obj}\" took {time.time() - obj_start_time} seconds")
                
            new_other = val['other'].copy()
            new_other['prompt'] = prompts
            
            update_new_data(new_data_json, new_key, new_img_path, new_keypoints, new_other)
            
            print(f"Generating {new_key} took {time.time() - curr_start_time} seconds") 
            print("************************")
            gen_img_time += (time.time() - curr_start_time)
            
            with open(f"{FLAGS.input_archive_path}data_jsonformat.json", 'w') as file:
                json.dump(data, file, indent=4)
                
            with open(f"{FLAGS.output_archive_path}data_jsonformat.json", 'w') as file:
                json.dump(new_data_json, file, indent=4)
    
    print(f"Generating images took {gen_img_time} seconds")
    
    # Update new data
    with open(f"{FLAGS.output_archive_path}data_jsonformat.json", 'w') as file:
        json.dump(new_data_json, file, indent=4)
    
    # Update old data
    with open(f"{FLAGS.input_archive_path}data_jsonformat.json", 'w') as file:
        json.dump(data, file, indent=4)
    
    print(f"Total time: {time.time() - start_time} seconds")
    
if __name__ == '__main__':
    app.run(main)