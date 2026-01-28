import cv2
import mediapipe as mp
import json
import numpy as np
import os
import argparse

def detect_keypoints(image_path, output_path):
    """
    Detects keypoints using MediaPipe and saves them in OpenPose JSON format.
    """
    mp_pose = mp.solutions.pose
    
    # Initialize MediaPipe Pose
    with mp_pose.Pose(
        static_image_mode=True,
        model_complexity=2,
        enable_segmentation=False,
        min_detection_confidence=0.5) as pose:

        image = cv2.imread(image_path)
        if image is None:
            print(f"Error: Could not read image {image_path}")
            return False

        # Convert the BGR image to RGB before processing.
        results = pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

        if not results.pose_landmarks:
            print(f"No body detected in {image_path}")
            return False

        # Convert MediaPipe landmarks to OpenPose BODY_25 format
        # MediaPipe has 33 landmarks. OpenPose BODY_25 has 25.
        # We need to map them manually. Mismatched points will be zeroed.
        
        # OpenPose BODY_25 Keypoints mapping:
        # 0: Nose, 1: Neck, 2: RShoulder, 3: RElbow, 4: RWrist,
        # 5: LShoulder, 6: LElbow, 7: LWrist, 8: MidHip, 9: RHip,
        # 10: RKnee, 11: RAnkle, 12: LHip, 13: LKnee, 14: LAnkle,
        # 15: REye, 16: LEye, 17: REar, 18: LEar, 19: LBigToe,
        # 20: LSmallToe, 21: LHeel, 22: RBigToe, 23: RSmallToe, 24: RHeel

        # MediaPipe Landmarks mapping (partial):
        # 0: nose, 
        # 11: left_shoulder, 12: right_shoulder, 
        # 13: left_elbow, 14: right_elbow, 
        # 15: left_wrist, 16: right_wrist, 
        # 23: left_hip, 24: right_hip, 
        # 25: left_knee, 26: right_knee, 
        # 27: left_ankle, 28: right_ankle
        # 2: left_eye, 5: right_eye
        # 7: left_ear, 8: right_ear
        # 29: left_heel, 30: right_heel
        # 31: left_foot_index, 32: right_foot_index (approximates for BigToe)
        
        h, w, _ = image.shape
        mp_lm = results.pose_landmarks.landmark
        
        keypoints = [0.0] * 75 # 25 points * (x, y, confidence)

        def get_point(idx):
            lm = mp_lm[idx]
            return [lm.x * w, lm.y * h, lm.visibility]

        # Helper to set point in OpenPose array
        def set_kp(op_idx, mp_idx):
            if mp_idx is not None:
                p = get_point(mp_idx)
                keypoints[op_idx*3] = p[0]
                keypoints[op_idx*3+1] = p[1]
                keypoints[op_idx*3+2] = p[2]

        # 0: Nose
        set_kp(0, 0)
        
        # 1: Neck - Average of shoulders in MP
        l_sh = get_point(11)
        r_sh = get_point(12)
        neck_x = (l_sh[0] + r_sh[0]) / 2
        neck_y = (l_sh[1] + r_sh[1]) / 2
        neck_conf = (l_sh[2] + r_sh[2]) / 2
        keypoints[1*3] = neck_x
        keypoints[1*3+1] = neck_y
        keypoints[1*3+2] = neck_conf

        # 2: RShoulder <- 12
        set_kp(2, 12)
        # 3: RElbow <- 14
        set_kp(3, 14)
        # 4: RWrist <- 16
        set_kp(4, 16)
        
        # 5: LShoulder <- 11
        set_kp(5, 11)
        # 6: LElbow <- 13
        set_kp(6, 13)
        # 7: LWrist <- 15
        set_kp(7, 15)

        # 8: MidHip - Average of hips
        l_hip = get_point(23)
        r_hip = get_point(24)
        mid_hip_x = (l_hip[0] + r_hip[0]) / 2
        mid_hip_y = (l_hip[1] + r_hip[1]) / 2
        mid_hip_conf = (l_hip[2] + r_hip[2]) / 2
        keypoints[8*3] = mid_hip_x
        keypoints[8*3+1] = mid_hip_y
        keypoints[8*3+2] = mid_hip_conf

        # 9: RHip <- 24
        set_kp(9, 24)
        # 10: RKnee <- 26
        set_kp(10, 26)
        # 11: RAnkle <- 28
        set_kp(11, 28)

        # 12: LHip <- 23
        set_kp(12, 23)
        # 13: LKnee <- 25
        set_kp(13, 25)
        # 14: LAnkle <- 27
        set_kp(14, 27)

        # 15: REye <- 5
        set_kp(15, 5)
        # 16: LEye <- 2
        set_kp(16, 2)
        # 17: REar <- 8
        set_kp(17, 8)
        # 18: LEar <- 7
        set_kp(18, 7)

        # 19: LBigToe <- 31
        set_kp(19, 31)
        # 20: LSmallToe - No direct map, leave 0 or approx? Leaving 0.
        
        # 21: LHeel <- 29
        set_kp(21, 29)

        # 22: RBigToe <- 32
        set_kp(22, 32)
        # 23: RSmallToe - No direct map.

        # 24: RHeel <- 30
        set_kp(24, 30)

        # Construct JSON
        output_data = {
            "version": 1.3,
            "people": [{
                "person_id": [-1],
                "pose_keypoints_2d": keypoints,
                "face_keypoints_2d": [],
                "hand_left_keypoints_2d": [],
                "hand_right_keypoints_2d": [],
                "pose_keypoints_3d": [],
                "face_keypoints_3d": [],
                "hand_left_keypoints_3d": [],
                "hand_right_keypoints_3d": []
            }]
        }

        with open(output_path, 'w') as f:
            json.dump(output_data, f)
        
        print(f"Keypoints saved to {output_path}")
        return True

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', type=str, required=True, help='Path to input image')
    parser.add_argument('--write_json', type=str, required=True, help='Path to output JSON')
    args = parser.parse_args()

    detect_keypoints(args.image, args.write_json)
