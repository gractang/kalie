import os
import base64
import requests
from io import BytesIO
import time

# Get OpenAI API Key from environment variable
api_key = os.environ['OPENAI_API_KEY']

# TODO(kuanfang): Maybe also support free form responses.
headers = {
    'Content-Type': 'application/json',
    'Authorization': f'Bearer {api_key}'
}

DEFAULT_LLM_MODEL_NAME = 'gpt-4'
DEFAULT_VLM_MODEL_NAME = 'gpt-4-vision-preview'


def encode_image_from_file(image_path):
    # Function to encode the image
    with open(image_path, 'rb') as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')


def encode_image_from_pil(image):
    buffered = BytesIO()
    image.save(buffered, format='JPEG')
    return base64.b64encode(buffered.getvalue()).decode('utf-8')


def prepare_payload(messages,
                   images,
                   meta_prompt,
                   model_name,
                   local_image,
                   temperature):

    user_content = []

    if not isinstance(messages, list):
        messages = [messages]

    for message in messages:
        content = {
            'type': 'text',
            'text': message,
        }
        user_content.append(content)

    if not isinstance(images, list):
        images = [images]

    for image in images:
        if local_image:
            base64_image = encode_image_from_file(image)
        else:
            base64_image = encode_image_from_pil(image)

        content = {
            'type': 'image_url',
            'image_url': {
                'url': f'data:image/jpeg;base64,{base64_image}'
            }
        }
        user_content.append(content)

    payload = {
        'model': model_name,
        'messages': [
            {
                'role': 'system',
                'content': [
                    {
                        'type': 'text',
                        'text': meta_prompt,
                    }
                ]
            },
            {
                'role': 'user',
                'content': user_content,
            }
        ],
        'max_tokens': 800,
        'temperature': temperature
    }
    return payload


def request_gpt(message,
                images,
                meta_prompt='',
                model_name=None,
                local_image=False,
                temperature=1):

    # TODO(kuan): A better interface should allow interleaving images and
    # messages in the inputs.

    if model_name is None:
        if images is [] or images is None:
            model_name = DEFAULT_LLM_MODEL_NAME
        else:
            model_name = DEFAULT_VLM_MODEL_NAME

    payload = prepare_payload(message,
                             images,
                             meta_prompt=meta_prompt,
                             model_name=model_name,
                             local_image=local_image,
                             temperature=temperature)

    attempts = 0
    while attempts < 5:
        attempts += 1
        try:
            response = requests.post(
                'https://api.openai.com/v1/chat/completions',
                headers=headers,
                json=payload)
            # Check the status code of the response
            if response.status_code == 200:
                return response.json()['choices'][0]['message']['content']
            else:
                print(f"Attempt {attempts}: Received invalid response: {response.status_code}")
        except requests.RequestException as e:
            print(f"Attempt {attempts}: Request failed with exception {e}")
            # print('\nInvalid response: ')
            # print(response)
            # print('\nInvalid response: ')
            # print(response.json())
        
        # Wait a bit before retrying
        time.sleep(2 ** attempts)  # Exponential backoff
        
    # If all attempts fail, raise an exception or handle it accordingly
    raise Exception("Failed to get a valid response after several attempts")


def prepare_payload_incontext(
        messages,
        images,
        meta_prompt,
        model_name,
        local_image,
        example_images,
        example_responses,
):

    user_content = []

    if not isinstance(messages, list):
        messages = [messages]

    for message in messages:
        content = {
            'type': 'text',
            'text': message,
        }
        user_content.append(content)

    if not isinstance(images, list):
        images = [images]

    for example_image, example_response in zip(
            example_images, example_responses):
        if local_image:
            base64_image = encode_image_from_file(example_image)
        else:
            base64_image = encode_image_from_pil(example_image)

        content = {
            'type': 'image_url',
            'image_url': {
                'url': f'data:image/jpeg;base64,{base64_image}'
            }
        }
        user_content.append(content)

        content = {
            'type': 'text',
            'text': example_response,
        }
        user_content.append(content)

    for image in images:
        if local_image:
            base64_image = encode_image_from_file(image)
        else:
            base64_image = encode_image_from_pil(image)

        content = {
            'type': 'image_url',
            'image_url': {
                'url': f'data:image/jpeg;base64,{base64_image}'
            }
        }
        user_content.append(content)

    payload = {
        'model': model_name,
        'messages': [
            {
                'role': 'system',
                'content': [
                    meta_prompt
                ]
            },
            {
                'role': 'user',
                'content': user_content,
            }
        ],
        'max_tokens': 800,
        'temperature': temperature
    }

    return payload


def request_gpt_incontext(
        message,
        images,
        meta_prompt='',
        example_images=None,
        example_responses=None,
        model_name=None,
        local_image=False):

    # TODO(kuan): A better interface should allow interleaving images and
    # messages in the inputs.

    if model_name is None:
        if images is [] or images is None:
            model_name = DEFAULT_LLM_MODEL_NAME
        else:
            model_name = DEFAULT_VLM_MODEL_NAME

    payload = prepare_payload_incontext(
        message,
        images,
        meta_prompt=meta_prompt,
        model_name=model_name,
        local_image=local_image,
        example_images=example_images,
        example_responses=example_responses)

    response = requests.post(
        'https://api.openai.com/v1/chat/completions',
        headers=headers,
        json=payload)

    try:
        res = response.json()['choices'][0]['message']['content']
    except Exception:
        print('\nInvalid response: ')
        print(response)
        print('\nInvalid response: ')
        print(response.json())
        exit()

    return res