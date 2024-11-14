from cv2 import aruco

# Robot Params #
nuc_ip = "172.16.0.3"
robot_ip = "172.16.0.2"
sudo_password = "Robots4good"
robot_type = 'fr3'
robot_serial_number = "290102-1324143"

# Camera ID's #
hand_camera_id = "13062452"
varied_camera_1_id = "24259877"
varied_camera_2_id = "20521388"

# Charuco Board Params #
CHARUCOBOARD_ROWCOUNT = 9
CHARUCOBOARD_COLCOUNT = 14
CHARUCOBOARD_CHECKER_SIZE = 0.020
CHARUCOBOARD_MARKER_SIZE = 0.016
ARUCO_DICT = aruco.Dictionary_get(aruco.DICT_5X5_100)

# Code Version [DONT CHANGE] #
r2d2_version = "1.3"
cvp_version = "1.0"
