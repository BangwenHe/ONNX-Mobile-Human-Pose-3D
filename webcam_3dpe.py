import cv2
import numpy as np
from imread_from_url import imread_from_url

from mobileHumanPose import MobileHumanPose, build_default_detector
from mobileHumanPose.utils_pose_estimation import draw_skeleton, draw_heatmap, vis_3d_multiple_skeleton

draw_detections = True
draw_3dpose = False # TODO: make 3d pl # Draw detected person bounding boxes 

# Camera parameters for the deprojection
# TODO: Correct the deprojection function to properly transform the joints to 3D
focal_length = [None, None]
principal_points = [None, None]

pose_model_path='models/mobile_human_pose_working_well_256x256.onnx'
pose_estimator = MobileHumanPose(pose_model_path, focal_length, principal_points)

# Initialize person detector
person_detector = build_default_detector(
    r"C:\Users\Bangwen\PycharmProjects\RealsenseDepth\nanodet-m.yml",
    r"C:\Users\Bangwen\PycharmProjects\RealsenseDepth\nanodet_m.ckpt"
)
detector_threshold = 0.5

# Initialize webcam
cap = cv2.VideoCapture(0)
cv2.namedWindow("Estimated pose", cv2.WINDOW_NORMAL)

skip_detection = False
while(True):
    ret, frame = cap.read()

    meta, boxes = person_detector.inference(frame)
    boxes = np.array(boxes[0][0])
    boxes = boxes[boxes[:, 4] > detector_threshold]

    # Skip pose estimation if no person has been detected
    if boxes is not None:

        # Simulate depth based on the bouding box area
        areas = (boxes[:,2] - boxes[:,0]) * (boxes[:,3] - boxes[:,1])
        depths = 500/(areas/(frame.shape[0]*frame.shape[1]))+500

        # Draw detected person bounding boxes 
        pose_img = frame.copy()

        # Initialize the represntation images 
        heatmap_viz_img = frame.copy()
        img_heatmap = np.empty(frame.shape[:2])
        pose_3d_list = []

        # Estimate the pose for each detected person
        for i, bbox in enumerate(boxes):
            
            # Draw the estimated pose
            keypoints, pose_3d, person_heatmap, scores = pose_estimator(frame, bbox, depths[i])
            pose_img = draw_skeleton(pose_img, keypoints, bbox[:2], scores)

            # Add the person heatmap to the image heatmap
            img_heatmap[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])] += person_heatmap 

            # Add the 3d pose to the list
            pose_3d_list.append(pose_3d)

        # Draw heatmap
        # heatmap_viz_img = draw_heatmap(heatmap_viz_img, img_heatmap)

        # Draw 3D pose
        # if draw_3dpose:
        #     vis_kps = np.array(pose_3d_list)
        #     img_3dpos = vis_3d_multiple_skeleton(vis_kps, np.ones_like(vis_kps))
        #     img_3dpos = cv2.resize(img_3dpos[200:-200,150:-150], frame.shape[1::-1])

        #     # Combine the images for showing them together
        #     combined_img = np.hstack((heatmap_viz_img, pose_img, img_3dpos))
        # else:

        #     # Combine the images for showing them together
        #     combined_img = np.hstack((heatmap_viz_img, pose_img))

        combined_img = cv2.resize(pose_img, (960, 720))
        cv2.imshow("Estimated pose", combined_img)

    else:
        # Show the same frame multiple times to avoid image size from changing
        if draw_3dpose:
            combined_img = np.hstack((frame, frame, frame))
        else:
            combined_img = np.hstack((frame, frame))

        print("No person was detected")
        cv2.imshow("Estimated pose", combined_img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break