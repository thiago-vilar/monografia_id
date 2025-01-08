import os
import cv2
import time
from robot_api.robot.robot import Robot
from vision_api.api_vision import VisionAPI
from vision_api.enums.classes_enum import CameraModel, RobotModel, MarkerModel
from packages.proto.pb.packages import MedicationsRelationWorld
from identify import DetectMedicine

def initial_positioning(robot_object, quadrant_number, pose_data):
    home = pose_data.reference_poses[13]
    quadrant_pose = next((pose for pose in pose_data.reference_poses if pose.description == f'quadrante{quadrant_number}'), None)
    robot_object.move_joints([home.position.x, home.position.y, home.position.z, home.angles.x, home.angles.y, home.angles.z])
    robot_object.move_joints([quadrant_pose.position.x, quadrant_pose.position.y, quadrant_pose.position.z, quadrant_pose.angles.x, quadrant_pose.angles.y, quadrant_pose.angles.z])

def capture_and_analyze_medicine(robot_object, marker_id, vision_api_object, quadrant_number):
    photo_directory = 'medicine_photos'
    if not os.path.exists(photo_directory):
        os.makedirs(photo_directory)

    # Move to initial position
    initial_positioning(robot_object, quadrant_number, vision_api_object.pose_data)

    # Detect and approach the marker
    medicine_marker = vision_api_object.get_marker_coordinates(marker_id)
    robot_object.move_to_marker(medicine_marker)

    # Capture image
    camera = cv2.VideoCapture(0)
    ret, frame = camera.read()
    if ret:
        image_path = os.path.join(photo_directory, f'medicine_{marker_id}.jpg')
        cv2.imwrite(image_path, frame)
        print(f'Image captured and saved at {image_path}')

        # Analyze captured image
        detector = DetectMedicine(image_path)
        detector.process_image()

    camera.release()

def main(robot, vision_api, marker_quadrant_mapping):
    for quadrant, markers in marker_quadrant_mapping.items():
        for marker in markers:
            capture_and_analyze_medicine(robot, marker, vision_api, quadrant)

if __name__ == "__main__":
    # Setup and connections omitted for brevity
    robot = Robot()
    robot.connect_with_ethernet()
    vision_api = VisionAPI(CameraModel.BRIO, RobotModel.KINOVA, MarkerModel.STAG)
    marker_quadrant_mapping = {
        1: [],
        2: [],
        3: [],
        4: [],
        5: [],
        6: [],
        7: [3],
        8: []
    }
    main(robot, vision_api, marker_quadrant_mapping)
