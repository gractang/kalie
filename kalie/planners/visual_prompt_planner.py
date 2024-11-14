import os
import pprint

import matplotlib.pyplot as plt
import json
from PIL import Image

from kalie.utils.visual_prompt_utils import request_plan

from kalie.planners.planner import Planner
from kalie.params import *
from kalie.utils.utils import * 

pp = pprint.PrettyPrinter(indent=4)

# keypoints_2d: {'grasp': [404, 158], 'function': [566, 228], 'target': [682, 226]}
# waypoints_2d: {'pre_contact': [[598, 226]], 'post_contact': [[688, 228]]}
OFFSET = 10

# --------------------------------
# Planner
# --------------------------------

class VisualPromptPlanner(Planner):
    def __init__(
        self,
        config=None,
        gradio_http=None, 
        camera_info=None,
        debug=False,
    ):
        """Initialize."""
        super().__init__(config=config, camera_info=camera_info, debug=debug)
        self.config = config
        self.debug = debug
        self.camera_info = camera_info
        self.gradio_http = gradio_http

    def preprocess_image(self, obs):
        """Preprocess the image.

        Args:
            obs: The current observation.

        Return:
            obs: The preprocessed observation.
        """
        crop = self.config.camera.planner.crop
        cropped_image = (
            obs['image_data'][crop[0]:crop[2], crop[1]:crop[3]].copy())
        # flip image vertically
        cropped_image = cropped_image[::-1, ::-1, :]
        return cropped_image

    def transform_points(self, context_2d):
        def transform(point):
            # flip vertically
            crop = self.config.camera.planner.crop
            crop_shape = (crop[2] - crop[0], crop[3] - crop[1])
            # image_shape = obs['image_data'].shape
            # point = np.array(point)
            point[1] = crop_shape[0] - point[1]
            point[0] = crop_shape[1] - point[0]
            # add crop
            point[0] += self.config.camera.planner.crop[1]
            point[1] += self.config.camera.planner.crop[0]
            return point

        # transform keypoints to the original image
        for key in context_2d['keypoints_2d'].keys():
            if context_2d['keypoints_2d'][key] is not None:
                kp = context_2d['keypoints_2d'][key].copy()
                context_2d['keypoints_2d'][key] = transform(kp)

        # transform waypoints to the original image
        for key in context_2d['waypoints_2d'].keys():
            wps = context_2d['waypoints_2d'][key]
            for i in range(len(wps)):
                if wps[i] is not None:
                    wps[i] = transform(wps[i].copy())

        return context_2d  

    def sample_context(self, obs, obs_img=None, t=0, task_name='sweeping', output_format='json'):
        if not obs_img:
            print("No observation image")
            processed_img = self.preprocess_image(obs)
            obs_img = Image.fromarray(processed_img).convert('RGB')

            plt.imshow(obs_img)
            plt.show()

        subtask_id = 'subtask_' + str(t)
        os.makedirs(
            self.config.log_dir + f'/{task_name}/{subtask_id}',
            exist_ok=True)
        obs_image_path = self.config.log_dir + f'/{task_name}/{subtask_id}/obs_image.png'  # NOQA
        obs_img.save(obs_image_path)

        def preprocess_image(obs_image_path):
            w, h = 450, 450
            orig_image = Image.open(obs_image_path).convert('RGB') 
            cropped_image, crop = crop_realtime_eval(orig_image, w, h)
            cropped_image_path = obs_image_path[:-4] + "_cropped" + obs_image_path[-4:]
            cropped_image.save(cropped_image_path)
            return cropped_image_path, crop

        cropped_image_path, crop = preprocess_image(obs_image_path)
        
        grasp_proposals = self.grasp_sampler.sample_grasp(
            obs['depth_data'],
            obs['depth_filtered'],
            self.camera_info['params'],
            crop=self.config.grasp_sampler.crop,
            num_samples=self.config.grasp_sampler.num_samples,
        )

        points = request_plan(self.gradio_http, cropped_image_path, task_name, output_format) 
        print(f"Points: {points}")

        for i in range(len(points)):
            points[i] = descale_point(points[i], 450, 450)

        plt.imshow(Image.open(cropped_image_path))
        plt.title("Points on 450 x 450 center crop")
        for point in points:
            plt.scatter(point[0], point[1]) 
        plt.show()

        grasp_keypoint = transform_original(points[0], crop)
        function_keypoint = transform_original(points[1], crop)
        target_keypoint = transform_original(points[2], crop)
        pre_contact_waypoints = [transform_original(points[3], crop)]
        post_contact_waypoints = [transform_original(points[4], crop)] 

        plt.imshow(Image.open(obs_image_path))
        plt.title("Points on planner crop")
        plt.scatter(grasp_keypoint[0], grasp_keypoint[1]) 
        plt.scatter(function_keypoint[0], function_keypoint[1]) 
        plt.scatter(target_keypoint[0], target_keypoint[1]) 
        plt.scatter(pre_contact_waypoints[0][0], pre_contact_waypoints[0][1]) 
        plt.scatter(post_contact_waypoints[0][0], post_contact_waypoints[0][1]) 
        plt.show()

        context_2d = {
            'keypoints_2d': {
                'grasp': grasp_keypoint,
                'function': function_keypoint,
                'target': target_keypoint,
            },
            'waypoints_2d': {
                'pre_contact': pre_contact_waypoints,
                'post_contact': post_contact_waypoints,
            },
            'target_euler': None,
            'grasp_proposals': grasp_proposals,
        }

        context_2d = self.transform_points(context_2d)

        self.plot_viz(context_2d, obs['image_data'])
        
        # save context keypoints and waypoints to json file in same directory as image
        context_path = self.config.log_dir + f'/{task_name}/{subtask_id}/context.json'
        with open(context_path, 'w') as f:
            # only save keypoints_2d and waypoints_2d of context_2d
            data_to_save = {k: v for k, v in context_2d.items() if k in ['keypoints_2d', 'waypoints_2d']}
            json.dump(data_to_save, f, indent=4)

        
        context = self.compute_context_3d(obs, context_2d)
        for k, v in context.items():
            print(f"{k}: {v}")  # display context

        return context
    

    def plot_viz(self, context_2d, img): 
        fig, ax = plt.subplots()
        ax.imshow(img)
        plt.title("Points on original image")


        print(context_2d)

        context_2d_points = [] 

        for category, sub_dict in context_2d.items():
            if category == "target_euler" or category == "grasp_proposals":
                continue
            for key, value in sub_dict.items():
                if type(value[0]) == list: 
                    for point in value: 
                        context_2d_points.append(point)
                else:
                    context_2d_points.append(value)
            
        for point, color, label in zip(context_2d_points, 
                                        ['red', 'blue', 'green', 'yellow', 'purple'], 
                                        ['Grasp', 'Function', 'Target', 'Pre-contact', 'Post-contact']):
            plt.scatter(point[0], point[1], c=color, label=label) 
        
        plt.show()
            
    

    def subtask_done(self, obs, t):
        """Reset for the new task/episode.

        Args:
            obs: The current observation.
            t: The time step index.

        Return:
            True if the full task is done, False otherwise.
        """
        return True