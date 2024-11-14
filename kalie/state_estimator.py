from kalie.vision.vild import Vild
from PIL import Image
from kalie.params import *
import matplotlib.pyplot as plt

class StateEstimator:

    def __init__(self, language_description, config): 
        # TODO: implement integration of text description of game
        self.config = config
    
    # not really sure what the difference between this and estimate is
    def reset(self, obs):
        return self.estimate(obs)
    
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
    
    # takes in obs, which is returned from r2d2.robot_env's get_observation() method,
    # and returns obs_image, which is a cropped version of the top-down camera observation
    def estimate(self, obs):
        processed_img = self.preprocess_image(obs)
        obs_image = Image.fromarray(processed_img).convert('RGB')
        return obs_image