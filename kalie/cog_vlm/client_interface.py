from gradio_client import Client, file
import os

def load_prompt(task_name, output_format):
    path = f'./prompts/{task_name}_{output_format}.txt'
    if os.path.isfile(path):
        with open(path, 'r') as f:
            prompt = f.read()
        return prompt 

def request_cogvlm(gradio_http, obs_image_path, task_name, output_format):
    client = Client(gradio_http)
    prompt = load_prompt(task_name, output_format)
    return client.predict(input_text=prompt, image_prompt=file(obs_image_path), api_name="/predict")