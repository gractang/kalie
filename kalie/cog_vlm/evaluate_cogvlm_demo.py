import os
import torch
import argparse
from functools import partial
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sat import mpu, get_args, get_tokenizer
from sat.training.deepspeed_training import training_main
from sat.helpers import print_rank0
from utils.cog_vlm_models import FineTuneTestCogVLMModel
from utils.cog_vlm_utils import llama2_text_processor, llama2_text_processor_inference, get_image_processor
from utils.utils import calculate_errors, store_pred_label_strs
import json
from datetime import datetime

with open("./kalie/cog_vlm/file_path_config.json") as f:
    file_path_config = json.load(f)
    EVAL_OUTPUT_DIR = file_path_config["EVAL_OUTPUT_DIR"] 
    FILE_NAME = f"{datetime.now().strftime('%Y-%m-%d-%H-%M')}.json"

def data_collator(examples):
    examples = [ex for ex in examples if len(ex) > 0] # drop {}
    for example in examples:
        for k in example:
            if isinstance(example[k], list):
                example[k] = torch.tensor(example[k])
            elif isinstance(example[k], np.ndarray):
                example[k] = torch.from_numpy(example[k])
    img_args = {}
    tmp_example = examples[0]
    for k in tmp_example['vision']:
        if type(tmp_example['vision'][k]) is torch.Tensor:
            img_args['vision_'+k] = torch.cat([example['vision'][k] for example in examples])
        else:
            img_args['vision_'+k] = example['vision'][k]
    for example in examples:
        example.pop('vision')
        if 'cross' in example:
            example.pop('cross')

    model_args = {}
    tmp_example = examples[0]
    for k in tmp_example:
        if type(tmp_example[k]) is torch.Tensor:
            model_args[k] = torch.cat([example[k] for example in examples])
        else:
            model_args[k] = tmp_example[k]
    model_args.update(img_args)
    return model_args

from collections import defaultdict

def broadcast_auto(data_dict):
    type2list = defaultdict(list)
    other = []
    for k in data_dict:
        if type(data_dict[k]) is torch.Tensor:
            type2list[data_dict[k].dtype].append(k)
        else:
            other.append(k)
    new_data = {}
    for k in type2list:
        new_data.update(mpu.broadcast_data(type2list[k], data_dict, k))
    for k in other:
        new_data[k] = data_dict[k]
    return new_data

def get_batch(data_iterator, args, timers):
    # Broadcast data.
    timers('data loader').start()
    if data_iterator is not None:
        data = next(data_iterator)
    else:
        data = None
    timers('data loader').stop()
    data_b = broadcast_auto(data)
    for k in data_b:
        if type(data_b[k]) is torch.Tensor and data_b[k].dtype is not torch.int32 and data_b[k].dtype is not torch.long:
            if args.fp16:
                data_b[k] = data_b[k].half()
            elif args.bf16:
                data_b[k] = data_b[k].bfloat16()
    return data_b

from torch.nn import CrossEntropyLoss
import numpy as np

from sat.model.mixins import CachedAutoregressiveMixin
from sat.generation.autoregressive_sampling import filling_sequence
from sat.generation.sampling_strategies import BaseStrategy, BeamSearchStrategy


def chat(model, tokenizer, tokens,
         max_length: int = 3000, num_beams=5, top_p=0.95, top_k=0, temperature=0.8, **kwargs):
    inputs = tokens.to(model.parameters().__next__().device)[0]
    seq = torch.cat(
        [inputs, torch.tensor([-1] * (max_length - len(inputs)), device=inputs.device)], dim=0
    )
    strategy = BaseStrategy(temperature=temperature, top_p=0.4, top_k=1, end_tokens=[tokenizer.eos_token_id])
    # strategy = BeamSearchStrategy(temperature=temperature, top_p=top_p, top_k=top_k, end_tokens=[tokenizer.eos_token_id],
    #                               num_beams=num_beams, consider_end=True)
    get_func = llama2_text_processor_inference.get_func(None, None, image_rope_mask=kwargs['image_rope_mask'])
    output = filling_sequence(
        model, seq,
        batch_size=1,
        strategy=strategy,
        get_masks_and_position_ids=get_func,
        **kwargs
    )[0]  # drop memory

    return output

from torch.nn import CrossEntropyLoss

def forward_step_eval(data_iterator, model, args, timers):
        
    def compute_metrics(eval_preds):
        preds, labels, device = eval_preds
        preds = preds.unsqueeze(0)
        if isinstance(preds, tuple):
            preds = preds[0]
        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
        if args.ignore_pad_token_for_loss:
            # Replace -100 in the labels as we can't decode them.
            labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        score_dict = {
            "acc": [],
            "acc_w/o_case": [],
            "avg_mse": [],
            "grasp_mse": [],
            "func_mse": [],
            "targ_mse": [],
            "prec_mse": [],
            "postc_mse": [],
            "grasp_abs_x": [],
            "func_abs_x": [],
            "targ_abs_x": [],
            "prec_abs_x": [],
            "postc_abs_x": [],
            "grasp_abs_y": [],
            "func_abs_y": [],
            "targ_abs_y": [],
            "prec_abs_y": [],
            "postc_abs_y": []
        }
        for pred, label in zip(decoded_preds, decoded_labels):
            print("*****************************")
            
            if args.rank == 0:
                print('pred', pred, 'label', label, flush=True)
            if pred == label:
                score_dict['acc'].append(1.)
            else:
                score_dict['acc'].append(0.)
            if pred.lower() == label.lower():
                score_dict['acc_w/o_case'].append(1.)
            else:
                score_dict['acc_w/o_case'].append(0.)
            
            mses_per_point, avg_mse, abs_per_point_x, abs_per_point_y = calculate_errors(pred, label, 'json')
            score_dict['avg_mse'].append(avg_mse)
            
            for k, v in mses_per_point.items():
                if k == "Grasp Point":
                    score_dict['grasp_mse'].append(v)
                elif k == "Function Point":
                    score_dict['func_mse'].append(v)
                elif k == "Target Point":
                    score_dict['targ_mse'].append(v)
                elif k == "Pre-contact Point":
                    score_dict['prec_mse'].append(v)
                elif k == "Post-contact Point":
                    score_dict['postc_mse'].append(v)
            

            for k, v in abs_per_point_x.items():
                if k == "Grasp Point":
                    score_dict['grasp_abs_x'].append(v)
                elif k == "Function Point":
                    score_dict['func_abs_x'].append(v)
                elif k == "Target Point":
                    score_dict['targ_abs_x'].append(v)
                elif k == "Pre-contact Point":
                    score_dict['prec_abs_x'].append(v)
                elif k == "Post-contact Point":
                    score_dict['postc_abs_x'].append(v)
                    
            for k, v in abs_per_point_y.items():
                if k == "Grasp Point":
                    score_dict['grasp_abs_y'].append(v)
                elif k == "Function Point":
                    score_dict['func_abs_y'].append(v)
                elif k == "Target Point":
                    score_dict['targ_abs_y'].append(v)
                elif k == "Pre-contact Point":
                    score_dict['prec_abs_y'].append(v)
                elif k == "Post-contact Point":
                    score_dict['postc_abs_y'].append(v)

            store_pred_label_strs(EVAL_OUTPUT_DIR, FILE_NAME, f"pred {pred} label {label}")

        for k, v in score_dict.items():
            score_dict[k] = float(np.mean(v))

        return score_dict

    # Get the batch.
    timers('batch generator').start()
    data_b = get_batch(
        data_iterator, args, timers)
    timers('batch generator').stop()
    
    labels = data_b.pop('labels')
    # loss computation
    logits = model(**data_b)[0]
    lm_logits = logits.to(torch.float32)
    shift_labels = labels[..., 1:].contiguous()
    shift_logits = lm_logits[..., -1-shift_labels.size(-1):-1, :].contiguous()
    # Flatten the tokens
    loss_fct = CrossEntropyLoss(ignore_index=-100)
    loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
    loss = loss.to(torch.float32)

    context_len = int(data_b['context_length'][0])
    tokens = data_b['input_ids'][:, :context_len]
    data_b['vision_expert_mask'] = data_b['vision_expert_mask'][:, :context_len]
    data_b['image_embed_mask'] = data_b['image_embed_mask'][:, :context_len]
    data_b['image_rope_mask'] = data_b['image_rope_mask'][:, :context_len]
    
    data_b.pop('input_ids')
    data_b.pop('attention_mask')
    data_b.pop('position_ids')
    
    qid = data_b.pop('question_id')

    model.add_mixin('auto-regressive', CachedAutoregressiveMixin())
    outputs = chat(model, tokenizer, tokens, **data_b)[0][context_len:]
    # print(outputs)
    model.del_mixin('auto-regressive')
    
    

    return torch.tensor(loss, device=outputs.device), {k: torch.tensor(v, device=outputs.device) for k, v in
                                                    compute_metrics(
                                                        (outputs.cpu(), labels.cpu(), outputs.device)).items()}

def forward_step(data_iterator, model, args, timers):
    """Forward step."""

    # Get the batch.
    timers('batch generator').start()
    data_b = get_batch(
        data_iterator, args, timers)
    labels = data_b.pop('labels')
    timers('batch generator').stop()
    logits = model(**data_b)[0]
    lm_logits = logits.to(torch.float32)
    # Shift so that tokens < n predict n
    shift_labels = labels[..., 1:].contiguous()
    shift_logits = lm_logits[..., -1-shift_labels.size(-1):-1, :].contiguous()
    # Flatten the tokens
    loss_fct = CrossEntropyLoss(ignore_index=-100)
    loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
    loss = loss.to(torch.float32)

    return loss, {'loss': loss}

from utils.cog_vlm_utils import AgentItemDataset
def create_dataset_function(image_processor, text_processor, path, args):
    data = None
    with open(path, "r") as fb:  # have a separate path for training, validation, and testing
        data = json.load(fb)
    dataset = AgentItemDataset(image_processor, text_processor, list(data.values()), False)
    return dataset

if __name__ == '__main__':
    py_parser = argparse.ArgumentParser(add_help=False)
    py_parser.add_argument('--max_length', type=int)
    py_parser.add_argument('--ignore_pad_token_for_loss', action='store_false')
    py_parser.add_argument("--version", type=str, default="chat", help='version to interact with')
    py_parser.add_argument("--from_pretrained", type=str, default="cogvlm-chat", help='pretrained ckpt')
    py_parser.add_argument("--local_tokenizer", type=str, default="lmsys/vicuna-7b-v1.5", help='tokenizer path')
    py_parser.add_argument("--vit_checkpoint_activations", action='store_true')
    py_parser = FineTuneTestCogVLMModel.add_model_specific_args(py_parser)
    known, args_list = py_parser.parse_known_args()
    args = get_args(args_list)
    args = argparse.Namespace(**vars(args), **vars(known))
    if args.use_qlora:
        args.device = 'cpu'

    model, args = FineTuneTestCogVLMModel.from_pretrained(args.from_pretrained, args, overwrite_args={'model_parallel_size': args.model_parallel_size} if args.model_parallel_size != 1 else {})
    if args.use_qlora and torch.cuda.is_available():
        model = model.to('cuda')
    from utils.cog_vlm_utils import llama2_tokenizer
    tokenizer = llama2_tokenizer(args.local_tokenizer, signal_type=args.version)
    image_processor = get_image_processor(args.eva_args["image_size"][0])
    text_processor = llama2_text_processor(tokenizer, args.max_length, args.image_length)

    training_main(args, model_cls=model, forward_step_function=forward_step, create_dataset_function=partial(create_dataset_function, image_processor, text_processor), collate_fn=data_collator, forward_step_eval=forward_step_eval)