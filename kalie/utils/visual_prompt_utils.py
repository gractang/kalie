import json
from kalie.cog_vlm.client_interface import * 
from kalie.utils.utils import print_fail

def parse_json_string(res):
    res = res[res.find('{'):] if res.find('{') != 0 else res
    data = json.loads(res)
    grasp_point = [int(pt) for pt in data["Grasp Point"]]
    function_point = [int(pt) for pt in data["Function Point"]]
    target_point = [int(pt) for pt in data["Target Point"]]
    pre_contact_point = [int(pt) for pt in data["Pre-contact Point"]]
    post_contact_point = [int(pt) for pt in data["Post-contact Point"]]
    return [grasp_point, function_point, target_point, pre_contact_point, post_contact_point]

def request_plan(gradio_http, image_path, task_name, output_format='json'):
    """Decompose the task into subtasks.
    """
    res = ''
    while True:  
        res = request_cogvlm(gradio_http, image_path, task_name, output_format)
        try: 
            plan_info = parse_json_string(res)
            return plan_info
        except:
            print_fail(f"Invalid response: {res}")