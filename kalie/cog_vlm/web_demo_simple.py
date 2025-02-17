"""
This script is a simple web demo of the CogVLM and CogAgent models, designed for easy and quick demonstrations.
For a more sophisticated user interface, users are encouraged to refer to the 'composite_demo',
which is built with a more aesthetically pleasing Streamlit framework.

Usage:
- Use the interface to upload images and enter text prompts to interact with the models.

Requirements:
- Gradio (only 3.x,4.x is not support) and other necessary Python dependencies must be installed.
- Proper model checkpoints should be accessible as specified in the script.

Note: This demo is ideal for a quick showcase of the CogVLM and CogAgent models. For a more comprehensive and interactive
experience, refer to the 'composite_demo'.
"""
import gradio as gr
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from PIL import Image
import torch
import time
from sat.model.mixins import CachedAutoregressiveMixin
from sat.mpu import get_model_parallel_world_size
from sat.model import AutoModel

from utils.cog_vlm_models import CogVLMModel
from utils.cog_vlm_utils import chat, llama2_tokenizer, llama2_text_processor_inference, get_image_processor, parse_response



DESCRIPTION = '''<h1 style='text-align: center'> <a href="https://github.com/THUDM/CogVLM">CogVLM / CogAgent</a> </h1>'''

NOTES = '<h3> This app is adapted from <a href="https://github.com/THUDM/CogVLM">https://github.com/THUDM/CogVLM</a>. It would be recommended to check out the repo if you want to see the detail of our model, CogVLM & CogAgent. </h3>'

MAINTENANCE_NOTICE1 = 'Hint 1: If the app report "Something went wrong, connection error out", please turn off your proxy and retry.<br>Hint 2: If you upload a large size of image like 10MB, it may take some time to upload and process. Please be patient and wait.'


AGENT_NOTICE = 'Hint 1: To use <strong>Agent</strong> function, please use the <a href="https://github.com/THUDM/CogVLM/blob/main/utils/utils/template.py#L761">prompts for agents</a>.'

GROUNDING_NOTICE = 'Hint 2: To use <strong>Grounding</strong> function, please use the <a href="https://github.com/THUDM/CogVLM/blob/main/utils/utils/template.py#L344">prompts for grounding</a>.'






model = image_processor = text_processor_infer = None

is_grounding = False

def process_image_without_resize(image_prompt):
    image = Image.open(image_prompt)
    # print(f"height:{image.height}, width:{image.width}")
    timestamp = int(time.time())
    file_ext = os.path.splitext(image_prompt)[1]
    filename_grounding = f"examples/{timestamp}_grounding{file_ext}"
    return image, filename_grounding

from sat.quantization.kernels import quantize

import argparse
def load_model(args): 
    try:
        world_size
    except NameError:
        world_size = 1
    
    model, model_args = AutoModel.from_pretrained(
        args.from_pretrained,
        args=argparse.Namespace(
        deepspeed=None,
        local_rank=0,
        rank=0,
        world_size=world_size,
        model_parallel_size=world_size,
        mode='inference',
        fp16=args.fp16,
        bf16=args.bf16,
        skip_init=True,
        use_gpu_initialization=True if (torch.cuda.is_available() and args.quant is None) else False,
        device='cpu' if args.quant else 'cuda'),
        overwrite_args={'model_parallel_size': world_size} if world_size != 1 else {}
    )

    model = model.half()
    model = model.eval()
    assert world_size == get_model_parallel_world_size(), "world size must equal to model parallel size for cli_demo!"

    language_processor_version = model_args.text_processor_version if 'text_processor_version' in model_args else args.version
    tokenizer = llama2_tokenizer(args.local_tokenizer, signal_type=language_processor_version)
    image_processor = get_image_processor(model_args.eva_args["image_size"][0])
    cross_image_processor = get_image_processor(model_args.cross_image_pix) if "cross_image_pix" in model_args else None

    if args.quant:
        quantize(model, args.quant)
        if torch.cuda.is_available():
            model = model.cuda()
    model.add_mixin('auto-regressive', CachedAutoregressiveMixin())

    text_processor_infer = llama2_text_processor_inference(tokenizer, args.max_length, model.image_length)

    return model, image_processor, cross_image_processor, text_processor_infer


def post(
        input_text,
        temperature,
        top_p,
        top_k,
        image_prompt,
        result_previous,
        hidden_image,
        state
        ):
    result_text = [(ele[0], ele[1]) for ele in result_previous]
    for i in range(len(result_text)-1, -1, -1):
        if result_text[i][0] == "" or result_text[i][0] == None:
            del result_text[i]
    print(f"history {result_text}")
    
    global model, image_processor, cross_image_processor, text_processor_infer, is_grounding

    # try:
    if True:
        with torch.no_grad():
            pil_img, image_path_grounding = process_image_without_resize(image_prompt)
            response, _, cache_image = chat(
                    image_path="", 
                    model=model, 
                    text_processor=text_processor_infer,
                    img_processor=image_processor,
                    query=input_text, 
                    history=None, 
                    cross_img_processor=cross_image_processor,
                    image=pil_img, 
                    max_length=3000, 
                    top_p=top_p, 
                    temperature=temperature,
                    top_k=top_k,
                    invalid_slices=text_processor_infer.invalid_slices if hasattr(text_processor_infer, "invalid_slices") else [],
                    no_prompt=False,
                    args=state['args']
            )
    # except Exception as e:
    #     print("error message", e)
    #     result_text.append((input_text, 'Timeout! Please wait a few minutes and retry.'))
    #     return "", result_text, hidden_image

    answer = response
    if is_grounding:
        parse_response(pil_img, answer, image_path_grounding)
        new_answer = answer.replace(input_text, "")
        result_text.append((input_text, new_answer))
        result_text.append((None, (image_path_grounding,)))
    else:
        result_text.append((input_text, answer))
    print(result_text)
    print('finished')
    return "", result_text, hidden_image


def main(args):
    global model, image_processor, cross_image_processor, text_processor_infer, is_grounding
    model, image_processor, cross_image_processor, text_processor_infer = load_model(args)
    is_grounding = 'grounding' in args.from_pretrained
    
    gr.close_all()

    temperature = 0.8
    top_p = 0.4
    top_k = 10
    from functools import partial
    def easy_submit(input_text, image_prompt, temperature, top_p, top_k):
        top_k = int(top_k)
        state = {"args": args}
        return post(input_text, temperature, top_p, top_k, image_prompt, "", "", state)[1][0][1]
    demo = gr.Interface(
        fn=easy_submit,
        inputs=[
            gr.Textbox(label='Input Text', placeholder='Please enter text prompt below and press ENTER.'),
            gr.Image(type="filepath", label="Image Prompt", value=None),
            gr.Number(value=0.8, label="Temperature", minimum=0, maximum=10),
            gr.Number(value=0.4, label="Top P", minimum=0, maximum=1),
            gr.Number(value=10, label="Top K(int)", minimum=0, maximum=100),],
        outputs="text"
    )
    # demo.queue(concurrency_count=10)
    demo.launch(share=True)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_length", type=int, default=3000, help='max length of the total sequence')
    parser.add_argument("--top_p", type=float, default=0.4, help='top p for nucleus sampling')
    parser.add_argument("--top_k", type=int, default=1, help='top k for top k sampling')
    parser.add_argument("--temperature", type=float, default=.8, help='temperature for sampling')
    parser.add_argument("--version", type=str, default="chat", choices=['chat', 'vqa', 'chat_old', 'base'], help='version of language process. if there is \"text_processor_version\" in model_config.json, this option will be overwritten')
    parser.add_argument("--quant", choices=[8, 4], type=int, default=None, help='quantization bits')

    parser.add_argument("--from_pretrained", type=str, default="/home/user/checkpoints/merged_model", help='pretrained ckpt') 
    parser.add_argument("--local_tokenizer", type=str, default="lmsys/vicuna-7b-v1.5", help='tokenizer path')
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--stream_chat", action="store_true")
    args = parser.parse_args()
    rank = int(os.environ.get('RANK', 0))
    world_size = int(os.environ.get('WORLD_SIZE', 1))
    args = parser.parse_args()   
    main(args)
