# import numpy as np
import os
import time  # NOQA
import matplotlib.pyplot as plt
from scipy import ndimage  # NOQA

from kalie.utils import depth_utils  # NOQA
from kalie.utils import visualization_utils  # NOQA
from kalie.utils.grasp_utils import Grasp2D
from kalie.planners.planner import Planner
from PIL import Image


class DataCollectorPlanner(Planner):

    def reset(self, obs, command=None):
        """Reset for the new task/episode.

        Args:
            obs: The initial observation.
            command: The command for the final task as a string.
        """

        self.done_clicking = None

        return

    def sample_context(self, obs, t=0, request_context=None):
        """Reset for the new task/episode.

        Args:
            obs: The current observation.

        Return:
            context: A dictionary that contains the context information for the
                current subtask. This will be part of the input to the
                low-level policy/motion planner.
        """
        print('Please sequentially click on the grasp point, function point, target point, pre-contact point, post-contact point, then once more anywhere to continue.')  # NOQA

        self.fig, self.ax = plt.subplots(ncols=2)
        self.fig.set_figheight(5)
        self.fig.set_figwidth(15)

        grasp_proposals = self.grasp_sampler.sample_grasp(
            obs['depth_data'],
            obs['depth_filtered'],
            self.camera_info['params'],
            crop=self.config.grasp_sampler.crop,
            num_samples=self.config.grasp_sampler.num_samples,
        )

        self._click_on_points(obs)

        grasp_keypoint = self.points[0]
        function_keypoint = self.points[1]
        target_keypoint = self.points[2]
        pre_contact_waypoints = [self.points[3]]
        post_contact_waypoints = [self.points[4]]

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
            'target_euler': None,  # TODO(fangchen)
            'grasp_proposals': grasp_proposals,
        }

        context = self.compute_context_3d(context_2d,
                                          obs['depth_filtered'],
                                          self.camera_info['params'],
                                          self.config)

        for k, v in context.items():
            print(f"{k}: {v}")  # display context

        return context

    def reset_seg_and_keypoints(self):
        pass

    def subtask_done(self, obs, t):
        """Reset for the new task/episode.

        Args:
            obs: The current observation.
            t: The time step index.

        Return:
            True if the current subtask is done (or timeout), False otherwise.
        """
        return True

    def _click_on_points(self, obs):  # NOQA
        print('Clicking on points.')
        self.points = []

        # Display the image
        self.ax[0].imshow(obs['image_data'])
        self.ax[1].imshow(obs['depth_data'])

        y1, x1, y2, x2 = self.config.camera.planner.crop

        self.ax[0].plot([x1, x1], [y1, y2], color='red')
        self.ax[0].plot([x2, x2], [y1, y2], color='red')
        self.ax[0].plot([x1, x2], [y2, y2], color='red')
        self.ax[0].plot([x1, x2], [y1, y1], color='red')

        self.ax[1].plot([x1, x1], [y1, y2], color='red')
        self.ax[1].plot([x2, x2], [y1, y2], color='red')
        self.ax[1].plot([x1, x2], [y2, y2], color='red')
        self.ax[1].plot([x1, x2], [y1, y1], color='red')

        annotation_info = [
            ['ro', 'grasp'],
            ['yo', 'function'],
            ['bo', 'target'],
            ['co', 'pre-contact'],
            ['co', 'post-contact'],
        ]

        self.done_clicking = False

        # Function to be called when the mouse is clicked
        def onclick(event):
            # Check if there are less than 5 points already clicked
            point_id = len(self.points)

            if point_id < 5:

                # Add the point to the list
                point = (event.xdata, event.ydata)

                if not (x1 < point[0] < x2 and y1 < point[1] < y2):
                    if point_id == 0:
                        print('Invalid grasp keypoint. Skip function keypoint. Next, select only target point, pre-contact point and post-contact point.')  # NOQA
                        self.points.append(None)  # grasp keypoint
                        self.points.append(None)  # function keypoint
                    elif point_id == 1:
                        print('Invalid function keypoint. Skip pre-contact and post-contact keypoint. Next, select only target point.')  # NOQA
                        self.points.append(None)  # function keypoint
                    else:
                        raise ValueError('Selected point %d out of the crop.' % (point_id))  # NOQA

                else:
                    self.points.append(point)

                    # Plot and annotate the point
                    self.ax[0].plot(
                        event.xdata,
                        event.ydata,
                        annotation_info[point_id][0])
                    self.ax[0].annotate(
                        annotation_info[point_id][1],
                        (event.xdata, event.ydata),
                        textcoords='offset points',
                        xytext=(0, 10),
                        ha='center')

                    if point_id in [3, 4]:
                        self.ax[0].plot(
                            [point[0], self.points[2][0]],
                            [point[1], self.points[2][1]],
                            'c-')

                # Redraw the canvas
                plt.draw()
                
            # Small change here-- asking for an extra point after the 5th point so that we can see
            # all points that were selected (including the 5th point)
            if point_id == 5:
                print('Points: ', self.points)
                print('All points received. (Click anywhere on the image to start the motion)')  # NOQA
                self.done_clicking = True

            if self.done_clicking:
                plt.close()

        # Connect the click event to the onclick function
        cid = self.fig.canvas.mpl_connect('button_press_event', onclick)  # NOQA

        plt.show()
        
    '''
    Copied shamelessly from Kuan's planner class, but with modifications to ignore grasp proposals
    '''
    def compute_context_3d(self,
        context,
        depth,
        camera_params,
        config,
        debug=False):

        def deprojection(point):
            x = point[0]
            y = point[1]
            z = depth[int(y), int(x)]
            hand_xyz = depth_utils.deproject_pixel_to_3d(
                z,
                point,
                camera_params['intrinsics'],
                camera_params['extrinsics'],
                is_worldframe=True)
            hand_xyz[2] = hand_xyz[2] + config.motion.grasp_z_offset
            return hand_xyz, z

        # keypoints are dict of points
        context['keypoints_3d'] = {}
        context['keypoints_depth'] = {}
        for k in context["keypoints_2d"].keys():
            point = context["keypoints_2d"][k]
            print(k, point)
            if point is None:
                context['keypoints_3d'][k] = None
                context['keypoints_depth'][k] = None
            else:
                (
                    context['keypoints_3d'][k],
                    context['keypoints_depth'][k],
                ) = deprojection(point)

        return context