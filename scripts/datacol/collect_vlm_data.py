'''
Collects data for the VLM task. Allows the user to click on the relevant keypoints, and stores the data into an hdf5 file.
''' 

import sys; sys.path.append("./")
import os  # NOQA
from absl import app, flags, logging  # NOQA

os.environ['OPENAI_API_KEY'] = "" # use your own key
from r2d2.robot_env import RobotEnv  # NOQA
from kalie.drivers.vlm_data_collector import VLMDataCollector
from kalie.planners.data_collector_planner import DataCollectorPlanner
from kalie.utils.utils import bcolors
from kalie.utils.config_utils import load_config

FLAGS = flags.FLAGS

flags.DEFINE_string('config', './configs/vlm.yaml', 'Location of config file')
flags.DEFINE_string('data_dir', './data', 'Directory to save the data.')
flags.DEFINE_string('task', 'pouring', 'Name of the task.')
flags.DEFINE_integer('seed', 42, 'Random seed.')
flags.DEFINE_integer('num_data_pts', 15, 'Number of data points to collect')
flags.DEFINE_string('filename', 'training_data', 'Filename under which to save hdf5 data')

flags.DEFINE_boolean('debug', False, 'Debugging mode.')
flags.DEFINE_boolean('fast', False, 'Skip confirmation.')
flags.DEFINE_boolean('reuse_segmasks', False, 'Using computed segmasks.')
flags.DEFINE_boolean('manual', True, 'Manually clicking the points.')
flags.DEFINE_boolean('use_center', False, 'Using the com as keypoints.')
flags.DEFINE_boolean('fix_angle', False, 'Fixing the grasping angle.')
flags.DEFINE_boolean('incontext', False, 'Using in-context examples.')
flags.DEFINE_boolean('save_incontext', False, 'Saving in-context examples.')

task_instructions = {
    'trash_sweeping':
        'Sweep the trash off the table with the brush.',

    'drawer_closing':
        'Close the drawer.',

    'towel_hanging':
        'Hang the cloth on the rack.',

    'unplugging':
        'Unplug the usb stick from the computer',

    'pouring':
        'Pour the small items out of the scooper and into the bowl'

}

def main(_):  # NOQA
    config = load_config(FLAGS.config)

    task_instruction = task_instructions[FLAGS.task]
    save_traj_dir = os.path.join(config.data_dir, FLAGS.task)

    print('Task: ', FLAGS.task)
    print('Task Instruction: ', task_instruction)
    print('Output Directory: ', save_traj_dir)

    if FLAGS.save_incontext:
        config.log_dir = './incontext/tmp'

    if FLAGS.incontext:
        config.prompt_name = 'visual_prompt_planner_incontext'

    if FLAGS.manual:
        planner = DataCollectorPlanner(
            config=config,
            debug=FLAGS.debug,
        )
        config.max_subtasks = 10
    else:
        print(bcolors.WARNING + 'No planner specified' + bcolors.ENDC)

    config.log_dir = './figure_making'
    
    env = RobotEnv(
        action_space="cartesian_position",
        camera_kwargs=dict(
            varied_camera=dict(
                image=True,
                depth=True,
                pointcloud=False,
                concatenate_images=False,
            ),
        ),
    )

    task_instruction = task_instructions[FLAGS.task]
    save_data_dir = os.path.join(FLAGS.data_dir, FLAGS.task)

    print('Task: ', FLAGS.task)
    print('Task Instruction: ', task_instruction)
    print('Output Directory: ', save_data_dir)

    data_collector = VLMDataCollector(
        env=env,
        planner=planner,
        config=config,
        task=FLAGS.task,
        debug=FLAGS.debug,
        skip_confirmation=FLAGS.fast,
        skip_reset=FLAGS.reuse_segmasks,
    )
    data_collector.collect_data(FLAGS.num_data_pts, FLAGS.filename)


if __name__ == '__main__':
    app.run(main)
