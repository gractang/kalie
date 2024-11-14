import numpy as np
from kalie.utils import depth_utils
from scipy import ndimage

def crop_image(image, crop):
        return image[crop[0]:crop[2], crop[1]:crop[3]]
    
class EnvironmentWrapper:
    def __init__(self, robot_env, language_description, config):
        self.env = robot_env
        self.language_description = language_description
        self.config = config
        
        self.wrist_camera_info = self.get_camera_info(
            self.config.camera.wrist)
        self.primary_camera_info = self.get_camera_info(
            self.config.camera.primary)
        self.secondary_camera_info = self.get_camera_info(
            self.config.camera.secondary)
        self.planner_camera_info = self.get_camera_info(
            self.config.camera.planner)
        
        self.control_hz = self.env.control_hz

    def get_planner_observation(self):
        """
        Returns the observation dict from the robot environment
        with some additional fields
        inspired by data collector's get_observation method
        """
        
        # The below is from get_planner_obs, trying to see the difference
        # between this and get_obs
        assert self.config.camera.depth.average_across_n_frames >= 1

        camera_name = self.planner_camera_info['name']
        camera_params = self.planner_camera_info['params']  # NOQA

        depth_data_list = []
        for i in range(self.config.camera.depth.average_across_n_frames):
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

        # Proprioception
        robot_state = raw_obs['robot_state']
        robot_pose = robot_state['cartesian_position']
        gripper_pose = robot_state['gripper_position']
        proprio_data = np.concatenate(
            [robot_pose, np.array([gripper_pose])])
        obs['proprio'] = proprio_data
        return obs
    
    def get_observation(self):
        """
        Like get_planner_observation but without the bells and whistles.
        Specifically for inside the execution loop.
        """
        obs = self.env.get_observation()
        del obs['depth']

        # Proprioception
        robot_state = obs['robot_state']
        robot_pose = robot_state['cartesian_position']
        gripper_pose = robot_state['gripper_position']
        proprio_data = np.concatenate(
            [robot_pose, np.array([gripper_pose])])
        obs['proprio'] = proprio_data
        obs['crop'] = dict()

        # print(
        #     'current ee position, ee pose, gripper position',
        #     robot_pose[:3],
        #     robot_pose[3:],
        #     gripper_pose,
        # )

        for camera_info in [self.wrist_camera_info,
                            self.primary_camera_info,
                            self.secondary_camera_info]:
            camera_name = camera_info['name']
            camera_crop = camera_info['crop']
            image_data = obs['image'][camera_name][:, :, :3][:, :, ::-1]
            image_data_cropped = crop_image(image_data, camera_crop)

            obs['image'][camera_name] = image_data_cropped
            obs['crop'][camera_name] = np.array(camera_crop)

        return obs

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
        
    def step(self, action):
        return self.env.step(action)
