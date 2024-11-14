from copy import deepcopy

import gym
import numpy as np

from r2d2.calibration.calibration_utils import load_calibration_info
from r2d2.camera_utils.info import camera_type_dict
from r2d2.camera_utils.wrappers.multi_camera_wrapper import (
    MultiCameraWrapper,
)
from r2d2.misc.parameters import hand_camera_id, nuc_ip
from r2d2.misc.server_interface import ServerInterface
from r2d2.misc.time import time_ms
from r2d2.misc.transformations import change_pose_frame


class RobotEnv(gym.Env):
    def __init__(
        self,
        action_space="cartesian_velocity",
        camera_kwargs={},
        reset=True,
        calibration_mode=False,
    ):
        # Initialize Gym Environment
        super().__init__()

        # Define Action Space #
        assert action_space in [
            "cartesian_position",
            "joint_position",
            "cartesian_velocity",
            "joint_velocity",
        ]
        self.action_space = action_space
        self.check_action_range = "velocity" in action_space
        self.calibration_mode = calibration_mode

        # Robot Configuration
        self.reset_joints = np.array(
            [0, -1 / 5 * np.pi, 0, -4 / 5 * np.pi, 0, 3 / 5 * np.pi, 0.0]
        )
        self.reset_joints_calibration = np.array(
            [0.2455860823392868, 0.6678642630577087, 0.11811266839504242, -1.9901493787765503, -2.3987948894500732, 2.3180248737335205, -0.4272650182247162]
        )  # reset to neutral pose

        # change the randomization range because of current camera position
        self.randomize_low = np.array(
            [-0.05, -0.05, -0.05, -0.05, -0.05, -0.05]
        )
        self.randomize_high = np.array(
            [0.05, 0.05, 0.05, 0.05, 0.05, 0.05]
        )
        self.DoF = 7 if ("cartesian" in action_space) else 8
        self.control_hz = 5  # TODO(kuan): Changed this for CVP.

        if nuc_ip is None:
            from franka.robot import FrankaRobot

            self._robot = FrankaRobot()
        else:
            self._robot = ServerInterface(ip_address=nuc_ip)

        # Create Cameras
        self.camera_reader = MultiCameraWrapper(camera_kwargs)
        self.calibration_dict = load_calibration_info()
        self.camera_type_dict = camera_type_dict

        # Reset Robot
        if reset:
            print('reset robot when environment is created')
            self.reset()

    def step(self, action):
        # Check Action
        assert len(action) == self.DoF
        if self.check_action_range:
            assert (action.max() <= 1) and (action.min() >= -1)

        # Update Robot
        action_info = self.update_robot(
            action, action_space=self.action_space
        )

        # Return Action Info
        return action_info

    def reset(self, randomize=False):
        self._robot.update_gripper(0, velocity=False, blocking=True)

        if randomize:
            noise = np.random.uniform(
                low=self.randomize_low, high=self.randomize_high
            )
        else:
            noise = None

        if self.calibration_mode:
            print('reset robot with calibration mode')
            self._robot.update_joints(
                self.reset_joints_calibration,
                velocity=False,
                blocking=True,
                cartesian_noise=noise,
            )
        else:
            print('reset robot without in normal mode')
            self._robot.update_joints(
                self.reset_joints,
                velocity=False,
                blocking=True,
                cartesian_noise=noise,
            )

    def reset_to_joint(self, joint_pos):
        self._robot.update_gripper(0, velocity=False, blocking=True)
        self._robot.update_joints(
            joint_pos,
            velocity=False,
            blocking=True,
            cartesian_noise=None,
        )

    def update_robot(
        self, action, action_space="cartesian_velocity", blocking=False
    ):
        action_info = self._robot.update_command(
            action, action_space=action_space, blocking=blocking
        )
        return action_info

    def create_action_dict(self, action):
        return self._robot.create_action_dict(action)

    def read_cameras(self):
        return self.camera_reader.read_cameras()

    def get_state(self):
        read_start = time_ms()
        state_dict, timestamp_dict = self._robot.get_robot_state()
        timestamp_dict["read_start"] = read_start
        timestamp_dict["read_end"] = time_ms()
        return state_dict, timestamp_dict

    def get_side_camera_extrinsics(self):
        # Adjust gripper camere by current pose
        extrinsics = deepcopy(self.calibration_dict)
        return extrinsics

    def get_camera_extrinsics(self, state_dict):
        # Adjust gripper camere by current pose
        extrinsics = deepcopy(self.calibration_dict)
        for cam_id in self.calibration_dict:
            if hand_camera_id not in cam_id:
                continue
            gripper_pose = state_dict["cartesian_position"]
            extrinsics[cam_id + "_gripper_offset"] = extrinsics[cam_id]
            extrinsics[cam_id] = change_pose_frame(
                extrinsics[cam_id], gripper_pose
            )
        return extrinsics

    def get_observation(self):
        obs_dict = {"timestamp": {}}

        # Robot State #
        state_dict, timestamp_dict = self.get_state()
        obs_dict["robot_state"] = state_dict
        obs_dict["timestamp"]["robot_state"] = timestamp_dict

        # Camera Readings #
        camera_obs, camera_timestamp = self.read_cameras()
        obs_dict.update(camera_obs)
        obs_dict["timestamp"]["cameras"] = camera_timestamp

        # Camera Info #
        obs_dict["camera_type"] = deepcopy(self.camera_type_dict)
        extrinsics = self.get_camera_extrinsics(state_dict)
        obs_dict["camera_extrinsics"] = extrinsics

        return obs_dict
