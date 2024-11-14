# KALIE: Fine-Tuning Vision-Language Models for Open-World Manipulation without Robot Data
[Grace Tang]()\*,
[Swetha Rajkumar]()\*,
[Yifei Zhou](https://yifeizhou02.github.io/)\,
[Homer Rich Walke](https://homerwalke.com/),
[Sergey Levine](https://people.eecs.berkeley.edu/~svlevine/)†,
[Kuan Fang](https://kuanfang.github.io/)†<br/>
\* equal contribution, † equal advising

[PDF](https://arxiv.org/pdf/2409.14066) | [arXiv](https://arxiv.org/abs/2409.14066) | [Website](https://kalie-vlm.github.io/)

![transition](assets/transition.gif)

# Installation
Set up a conda environment:
```
conda create -n kalie python=3.10
conda activate kalie 
```

Clone the respository, using the `--recurse-submodules` flag in order to retrieve our modified Stable Diffusion and ControlNet repositories.
```
git clone --recurse-submodules https://github.com/gractang/kalie.git
```

Install [GroundingDINO](https://github.com/IDEA-Research/GroundingDINO) following their instructions.

Install [SAM](https://github.com/facebookresearch/segment-anything):
```
pip install git+https://github.com/facebookresearch/segment-anything.git
```

Install [Detectron2](https://detectron2.readthedocs.io/en/latest/tutorials/install.html):

```
python -m pip install 'git+https://github.com/facebookresearch/detectron2.git'
```

If you run into any issues, please refer to the original repositories for more detailed instructions.

Download the DINO and SAM checkpoints in the repository.

```
cd kalie
mkdir models && cd models
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
wget https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth
```

Install the dependencies for [CogVLM](https://github.com/IDEA-Research/GroundingDINO) following their instructions.

```
pip install wandb
pip install gradio
```

### Setting up ControlNet
In order to use custom controls, download the `.pth` models of your choice from the [ControlNet Hugging Face repository](https://huggingface.co/lllyasviel/ControlNet-v1-1/tree/main) and place them into the `stable-diffusion-webui/extensions/sd-webui-controlnet/models` directory. In `scripts/datagen/configs`, we provide default configuration JSON files for the Canny, Depth Map, MLSD, Scribble, Segmentation Mask, and SoftEdge preprocessor controls, but it is straightforward to adjust them to suit individual needs.


# Usage

## Collecting Data
We provide scripts to collect data using the Franka robot and [DROID](https://droid-dataset.github.io/) platform, though for the purposes of data synthesis, any method of annotating images with keypoints such that the following file structure exists will work.

```
data/
|-- imgs/
    |-- entry_0.png
    |-- entry_1.png
    | ...
|-- data_jsonformat.json
```

Note that `data_jsonformat.json` must contain data of the form
```
{
    "entry_i": {
        "img": "<img_path>",
        "points": "{\n    \"Grasp Point\": [\n        <x1>,\n        <y1>\n    ],\n    \"Function Point\": [\n        <x2>,\n        <y2>\n    ],\n    \"Target Point\": [\n        <x3>,\n        <y3>\n    ],\n    \"Pre-contact Point\": [\n        <x4>,\n        <y4\>n    ],\n    \"Post-contact Point\": [\n        <x5>,\n        <y5>\n    ]\n}",
        "other": {
            "task": "<task_prompt>",
            "objects": [
                "<task_obj_1>",
                "<task_obj_2>", 
                ...
            ]
        }
    },
    ...
}
```
To collect data using our code, run
`python scripts/datacol/collect_vlm_data.py` from the root `kalie` directory. This generates an hdf5 file in the specified directory. Then, run `python scripts/datacol/hdf5_to_json.py --task "<task_prompt>" --filepath "<hdf5 filepath>" --outdir "<ouptut directory>` to convert it to a directory of the form above.

## KALIE Base Data 
The [datasets](https://huggingface.co/datasets/sr2022/KALIE_Original_Images) containing the human-annotated data points we collected for the sweeping, drawer closing, trowel pouring, towel hanging, and unplugging tasks can be accessed here.

## Generating Synthetic Data
First, navigate into the `stable-diffusion-webui/` repository, and run
```
./webui.sh --no-half --api
```
Then, change relevant parameters in `scripts/datagen/data_generation.py` and run the script from the root directory. An example usage is shown below:
```
python scripts/datagen/data_generation.py --input_archive_path "./input_archive" --num_prompts 10 --num_gpt_prompts 15
```
This command will pull data from `./input_archive` and ask the LLM for 15 prompts. Out of those, it will randomly select 10 prompts to generate images from. For an original dataset size of `N`, the final size of the dataset generated from the above command will be `11 * N`, (`10 * N` synthetic and the `N` original points).

## Fine Tuning CogVLM

Navigate to ```./kalie/cog_vlm/file_path_config.json``` and update the paths for training data, validation data, prompt, and evaluation output. Prompts used for tasks in KALIE can be seen in ```prompts/```

Then run  
```
bash kalie/cog_vlm/finetune_cogvlm_lora.sh
```
to run the fine-tuning script with language.

Merge the model by running 
```
bash scripts/merge_model.sh 
``` 
changing the ```--from_pretrained``` flag to the trained model checkpoint. 

Evaluate the model by running 
```
bash kalie/cog_vlm/evaluate_cogvlm.sh
```
changing the ```--from_pretrained``` flag to the merged model checkpoint. Predictions should appear in the directory as configured in ```./kalie/cog_vlm/file_path_config.json```

Start the CogVLM web server by running 
```
python kalie/cog_vlm/web_demo_simple.py
```
changing the ```--from_pretrained``` flag to the merged model checkpoint. 

Run real-time inference by calling ```request_cogvlm``` from ```./kalie/cog_vlm/client_interface.py``` passing in the public gradio url obtained from starting the web server. 


# Acknowledgements

We would like to thank Sudeep Dasari, Ria Doshi, Stefanie Gschwind, Fangchen Liu, Cyprien Noel, Karl Pertsch, and Paul Zhou for valuable supports with the infrastructure.

# BibTeX

```
@misc{tang2024kalie,
      title={KALIE: Fine-Tuning Vision-Language Models for Open-World Manipulation without Robot Data}, 
      author={Grace Tang and Swetha Rajkumar and Yifei Zhou and Homer Rich Walke and Sergey Levine and Kuan Fang},
      year={2024},
      eprint={2409.14066},
      archivePrefix={arXiv},
      primaryClass={cs.RO}
}
```

