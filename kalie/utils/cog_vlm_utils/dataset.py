import logging
from PIL import Image
from torch.utils.data import Dataset
from sat.helpers import print_rank0
import os 
from utils.utils import * 

class AgentItemDataset(Dataset):
    def __init__(self, image_processor, text_processor, data, is_training, cross_image_processor=None):
        super().__init__()
        self.data = data
        self.image_processor, self.text_processor, self.cross_image_processor = image_processor, text_processor, cross_image_processor
        self.is_training = is_training 
        with open("./kalie/cog_vlm/file_path_config.json") as f:
            file_path_config = json.load(f)
            self.TASK_PROMPT = self.load_prompt(file_path_config["PROMPT_PATH"])
        self.w, self.h = 450, 450 
        self.min_size_ratio = 0.95
        print("Prompt:")
        print(self.TASK_PROMPT)
        print_rank0(f"find {len(self.data)} samples in all...")
    
    def process_img(self, img):
        img_dict = {'vision': self.image_processor(img)}
        if self.cross_image_processor:
            img_dict.update({'cross': self.cross_image_processor(img)})
        return img_dict
    
    def load_prompt(self, prompt_path):
        if os.path.isfile(prompt_path) and prompt_path[-4:] == '.txt':
            with open(prompt_path, 'r') as f:
                prompt = f.read()
            return prompt 
    
    def process_text(self, answer, prompt):
        return self.text_processor(answer, prompt)
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        data = self.data[index]
        img_dict, label = None, None
        try:
            orig_image = Image.open("./" + data['img']).convert('RGB') 
            orig_points_dict = json_to_points_dicts(data["points"]) 
            orig_points = list(orig_points_dict.values()) 
            for i in range(len(orig_points)):
                orig_points[i] = descale_point(orig_points[i], orig_image.size[0], orig_image.size[1]) 

            if self.is_training: 
                cropped_image, cropped_points = get_image_transformation(orig_image, orig_points, self.w, self.h, self.min_size_ratio)
            else: 
                cropped_image, cropped_points = crop_evaluation_data(orig_image, orig_points, self.w, self.h)

            for i in range(len(cropped_points)):
                cropped_points[i] = scale_point(cropped_points[i], self.w, self.h)

            points_dict = {
                "Grasp Point": cropped_points[0],
                "Function Point": cropped_points[1],
                "Target Point": cropped_points[2],
                "Pre-contact Point": cropped_points[3],
                "Post-contact Point": cropped_points[4]
            }

            def tune_num(number):
                if number < 0:
                    return "000"
                elif number >= 1000:
                    return "999"
                return f"{number:03d}"  # format to 3 digits with leading zeros
            
            # Apply the function to each value in the dictionary
            new_points_dict = {key: (tune_num(value[0]), tune_num(value[1])) for key, value in points_dict.items()}

            # Convert the dictionary to a JSON string with indentation
            label = json.dumps(new_points_dict, indent=4)
            label = f"""{label}"""

            img_dict = self.process_img(cropped_image)
            
        except Exception as e:
            print_rank0(e, level=logging.WARNING)
            print("image path is inaccessible") 
            return {}

        text_dict = self.process_text(label, self.TASK_PROMPT)
        assert "labels" in text_dict, "No label found in text_dict."
        if text_dict is None:
            print_rank0(f"Process text failed. Please check the max_target_length & max_source_length.\n The data is {data}", level=logging.WARNING)
            return {}
        ret = {**img_dict, **text_dict, "question_id": data['img']}
        return ret
