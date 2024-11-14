import os
import time
from datetime import date
import h5py
import numpy as np
import matplotlib.pyplot as plt  # NOQA
from matplotlib import widgets  # NOQA
from scipy import ndimage

from kalie.utils import depth_utils  # NOQA

# Prepare Data Folder #
dir_path = os.path.dirname(os.path.realpath(__file__))
data_dir = os.path.join(dir_path, "../../vlm_data")

class VLMDataCollector:
    '''
    Collects data to finetune CogVLM
    Goal / Overview:
    - Show image and click sequentially on:
        - grasp point (where the object should be picked up; e.x. the handle of the broom)
        - function point (where the object in hand makes contact with other object; 
                          e.x. the hammer head, the brush of the broom.)
        - target point (where the piece should be placed)
        - pre-contact point (where the gripper should move to before grasping)
        - post-contact point (where the gripper should move to after target)
    - Inputs to the data collector:
        - Task instruction / language description (e.x. "pick up the broom and sweep the table")
        - Image of the board
    - Outputs (as a list of the dictionaries in hdf5 format):
        - Input task instruction / language description 
            - Note that we don't really need to store this for the one-task case
            - So for now we just won't store it
        - Image of the board
        - 5 keypoints (grasp, function, target, pre-contact, post-contact)
    '''
    def __init__(
        self,
        env,
        planner,
        config,
        task=None,
        debug=False,
        skip_confirmation=False,
        skip_reset=False,
    ):
        self.env = env
        self.planner = planner
        self.task = task

        self.config = config

        self.wrist_camera_info = self.get_camera_info(
            self.config.camera.wrist)
        self.primary_camera_info = self.get_camera_info(
            self.config.camera.primary)
        self.secondary_camera_info = self.get_camera_info(
            self.config.camera.secondary)
        self.planner_camera_info = self.get_camera_info(
            self.config.camera.planner)

        self.env_obs = None
        self.resume_joint_pos = None

        assert config.camera.planner.view == 'left', (
            'The depth image is only available from the left view.')
        self.planner.camera_info = self.planner_camera_info

        # Get Camera Info #
        self.cam_ids = list(env.camera_reader.camera_dict.keys())
        self.cam_ids.sort()

        self.debug = debug
        self.skip_confirmation = skip_confirmation
        self.skip_reset = skip_reset

        self.saved_env_obs = [None] * 10

    def get_camera_info(self, camera_config):
        camera_name = camera_config.serial_number + '_' + camera_config.view
        camera = self.env.camera_reader.get_camera(camera_config.serial_number)
        camera_intrinsics = camera.get_intrinsics()
        camera_extrinsics = self.env.get_side_camera_extrinsics()
        camera_params = {
            'intrinsics': camera_intrinsics[camera_name],
            'extrinsics': camera_extrinsics[camera_name],
        }
        return {
            'name': camera_name,
            'params': camera_params,
            'crop': camera_config.crop,
        }

    # Gets observations that are useful for the planner   
    def get_planner_observation(self, check=False):

        def show_image():
            if check:
                fig, ax = plt.subplots(ncols=2)
                fig.set_figheight(5)
                fig.set_figwidth(15)
                ax[0].axis('off')
                ax[1].axis('off')
                ax_button1 = plt.axes([0.25, 0.025, 0.1, 0.075])
                ax_button2 = plt.axes([0.65, 0.025, 0.1, 0.075])

            def on_ok_button_clicked(event):
                plt.close()

            def plot():
                print('Taking images, please wait...')
                self.env_obs = self._get_planner_observation()

                if check:
                    ax[0].imshow(self.env_obs['image_data'])
                    ax[1].imshow(self.env_obs['depth_data'])

                    button_ok = widgets.Button(ax_button1, "OK")
                    button_ok.on_clicked(
                        on_ok_button_clicked)

                    button_retake = widgets.Button(ax_button2, "Retake")
                    button_retake.on_clicked(
                        on_retake_button_clicked)

                    plt.show()

                print('Images updated.')

            # Define button click action
            def on_retake_button_clicked(b):
                plot()

            plot()

        show_image()

    def _get_planner_observation(self):
        assert self.config.camera.depth.average_across_n_frames >= 1

        camera_name = self.planner_camera_info['name']
        camera_params = self.planner_camera_info['params']  # NOQA

        depth_data_list = []
        for _ in range(self.config.camera.depth.average_across_n_frames):
            raw_obs = self.env.get_observation()

            depth_data = raw_obs['depth'][camera_name]
            assert depth_data.size > 0, (
                f"Invalid depth shape: {depth_data.shape}")
            depth_data_list.append(depth_data)

        obs = dict()

        obs['camera_intrinsics'] = camera_params['intrinsics']
        obs['camera_extrinsics'] = raw_obs['camera_extrinsics']

        depth_data = depth_utils.preprocess_depth_list(depth_data_list)
        obs['depth_data'] = depth_data

        depth_inpainted = depth_utils.inpaint(depth_data)
        depth_filtered = ndimage.gaussian_filter(
            depth_inpainted,
            sigma=self.config.camera.depth.grad_gaussian_sigma)
        obs['depth_filtered'] = depth_filtered

        # RGB image
        image_data = raw_obs['image'][camera_name][:, :, :3]
        image_data = image_data[:, :, ::-1]  # BGR -> RGB
        obs['image_data'] = image_data

        return obs
    
    def write_image_to_hdf5(self, filepath, init_image, datapoint_count):
        if datapoint_count == 0:
            with h5py.File(filepath, 'w') as hdf:
                # Create a group for the first entry
                grp = hdf.create_group('entry_0')
                grp.create_dataset('init_image', data=init_image, dtype=np.uint8)
        else:
            with h5py.File(filepath, 'a') as hdf:
                entry_name = f'entry_{datapoint_count}'
                grp = hdf.create_group(entry_name)
                
                # Create a dataset for the new image data
                grp.create_dataset('init_image', data=init_image, dtype=np.uint8)


    def preprocess_image(self, obs):
        crop = self.config.camera.planner.crop
        cropped_image = (
            obs['image_data'][crop[0]:crop[2], crop[1]:crop[3]].copy())
        # flip image vertically
        cropped_image = cropped_image[::-1, ::-1, :]
        return cropped_image
    
    def write_points_to_hdf5(self, filepath, context, datapoint_count):

        def transform(point):
            crop = self.config.camera.planner.crop
            crop_shape = (crop[2] - crop[0], crop[3] - crop[1])

            point = list(point)
            
            point[0] -= crop[1]
            point[1] -= crop[0]
            point[0] = crop_shape[1] - point[0]
            point[1] = crop_shape[0] - point[1]
            
            return point

        def coordinate_scale(point):
            crop = self.config.camera.planner.crop
            crop_shape = (crop[2] - crop[0], crop[3] - crop[1])
            max_x, max_y = crop_shape
            scaled_x = round(((point[0]) / (max_x)) * 1000)
            scaled_y = round(((point[1]) / (max_y)) * 1000)
            return scaled_x, scaled_y

        data_dict = {"grasp_point": coordinate_scale(transform(context['keypoints_2d']['grasp'])),
                     "function_point": coordinate_scale(transform(context['keypoints_2d']['function'])),
                     "target_point": coordinate_scale(transform(context['keypoints_2d']['target'])),
                     "pre_contact_point": coordinate_scale(transform(context['waypoints_2d']['pre_contact'][0])),
                     "post_contact_point": coordinate_scale(transform(context['waypoints_2d']['post_contact'][0]))}
        
        with h5py.File(filepath, 'a') as hdf:
            entry_name = f'entry_{datapoint_count}'
            
            # Access the group for this entry
            grp = hdf[entry_name]
            
            # Create datasets within this group for the new data
            grp.create_dataset('grasp_point', data=data_dict['grasp_point'])
            grp.create_dataset('function_point', data=data_dict['function_point'])
            grp.create_dataset('target_point', data=data_dict['target_point'])
            grp.create_dataset('pre_contact_point', data=data_dict['pre_contact_point'])
            grp.create_dataset('post_contact_point', data=data_dict['post_contact_point'])
    
    def preview_imgs(self):
        fig, ax = plt.subplots()
        
        ax.imshow(self.env_obs['image_data'], interpolation='nearest')
        
        y1, x1, y2, x2 = self.config.camera.planner.crop
        
        ax.plot([x1, x1], [y1, y2], color='red')
        ax.plot([x2, x2], [y1, y2], color='red')
        ax.plot([x1, x2], [y2, y2], color='red')
        ax.plot([x1, x2], [y1, y1], color='red')
        
        plt.show() 

    '''
    Calls data collector planner to click on the points, then writes the results
    and the image observation to the hdf5 file.
    '''
    def collect_data(self, num_trials, custom_name=None):
        os.makedirs(f"{self.config.data_dir}/{self.task}/", exist_ok=True)
        if custom_name is None:
            filepath = f"{self.config.data_dir}/{self.task}/{date.today()}_{num_trials}.hdf5"
        else:
            filepath = f"{self.config.data_dir}/{self.task}/{custom_name}_{num_trials}.hdf5"
        
        obs_so_far = [] # List of observations so far, used for doing all data collection first then point clicking after
        for i in range(num_trials):
            print(f"Trial {i}")
            self.get_planner_observation() # Updates self.env_obs to contain image and depth data
            self.preview_imgs()
            
            next = input("Successful? y/n")
            
            while next != 'y':
                print(f"Trial {i}")
                self.get_planner_observation()
                self.preview_imgs()
                next = input("Successful? y/n")
            
            crop = self.preprocess_image(self.env_obs)
            self.write_image_to_hdf5(filepath, crop, i) 
            obs_so_far.append(self.env_obs)
            
        for i in range(num_trials):
            print(f"Trial {i}")
            context = self.planner.sample_context(obs_so_far[i], t=i)
            next = input("Successful? y/n")
            while next != 'y':
                print(f"Trial {i}")
                context = self.planner.sample_context(obs_so_far[i], t=i)
                next = input("Successful? y/n")
            
            self.write_points_to_hdf5(filepath, context, i)
            