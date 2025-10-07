import cv2, csv, time, sys
import mediapipe as mp
import numpy as np
import math
import pyautogui

screen_w, screen_h = pyautogui.size()  


sys.path.append(r"D:\Coding")





#TODO make these a class ffs helps a bit with computation speed

def compute_angles(coordinate1, coordinate2, coordinate3):

    ax, ay = coordinate1
    bx, by = coordinate2
    cx, cy = coordinate3
    ab = (ax - bx, ay - by)
    cb = (cx-bx, cy-by)
    dot = ab[0]*cb[0] + ab[1]*cb[1]
    norm_ab = math.hypot(*ab)
    norm_cb = math.hypot(*cb)
    if norm_ab == 0 or norm_cb == 0:
        return None
    cosine = dot / (norm_ab * norm_cb)
    cosine = max(-1.0, min(1.0, cosine))

    return math.degrees(math.acos(cosine))    

def compute_coordinate_averages(*args):

    if not args:
        return None, None

    x_sum = 0
    y_sum = 0
    n = len(args)
    
    for point in args:
        x_sum += point[0]
        y_sum += point[1]
        
    return int(x_sum / n), int(y_sum / n)

def pad_borders(frame):
            # Padded Border
        h, w = frame.shape[:2]
        scale = min(screen_w / w, screen_h / h)
        new_w, new_h = int(w * scale), int(h * scale)
        resized = cv2.resize(frame, (new_w, new_h))
        
        top = (screen_h - new_h) // 2
        bottom = screen_h - new_h - top
        left = (screen_w - new_w) // 2
        right = screen_w - new_w - left
        
        bordered_frame = cv2.copyMakeBorder(resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[0, 0, 0])
        return bordered_frame        

def vfra(path, *args):
    """ Video Frame Realtime Analysis 
    (function) def vfra(video: String) -> None
    """
        
    visualize_angles=1
    compute_frames=1
    frame_counter = 0
    success_counter = 0
    loss_counter = 0

    video = cv2.VideoCapture(path)
    fps = video.get(cv2.CAP_PROP_FPS) if video.get(cv2.CAP_PROP_FPS)>0 else 30

    frame_delay = int(1000/fps)
    frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    print(frame_count)


        
    mp_drawing = mp.solutions.drawing_utils
    mp_pose = mp.solutions.pose

    cv2.namedWindow("Pose Analysis", cv2.WINDOW_NORMAL)
    cv2.setWindowProperty("Pose Analysis", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)





    pose = mp_pose.Pose(static_image_mode=False,
                        model_complexity=2 if args ==2 else (1 if args ==1 else 0),
                        enable_segmentation=False,
                        min_detection_confidence=0.5,
                        min_tracking_confidence=0.5,
                        # smooth_landmarks=True
                        )

    while video.isOpened():

        ret, frame = video.read()
        if not ret:
            break

        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(image_rgb)
        
        if results.pose_landmarks and compute_frames==1:
            mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS, 
                                      mp.solutions.drawing_utils.DrawingSpec(color=(0,0,255), thickness=2, circle_radius=1), #landmark
                                      mp.solutions.drawing_utils.DrawingSpec(color=(200,200,200), thickness=2) #connection
                                    )
            success_counter +=1
        elif results.pose_landmarks and compute_frames==0:
            print(f"landmarks skipped at {str(time.ctime())} [{frame_counter}]")
        else: 
            print(f"landmarks lost at {str(time.ctime())} [{frame_counter}]")
            loss_counter +=1
            
        if visualize_angles == 1:
            # Convert landmarks into coordinates
            h, w, _ = frame.shape
            textplacement = (200,500)
            if results.pose_landmarks:
                landmark_px_coordinates = [(int(lm.x * w), int(lm.y * h)) for lm in results.pose_landmarks.landmark]
                try:

                    right_arm_angle = round(compute_angles(landmark_px_coordinates[12], landmark_px_coordinates[14], landmark_px_coordinates[16]), 0)
                    left_arm_angle = round(compute_angles(landmark_px_coordinates[11], landmark_px_coordinates[13], landmark_px_coordinates[15]), 0)
                except TypeError:
                    pass
                right_arm_annotation_position = compute_coordinate_averages(landmark_px_coordinates[12], landmark_px_coordinates[14], landmark_px_coordinates[16])
                left_arm_annotation_position = compute_coordinate_averages(landmark_px_coordinates[11], landmark_px_coordinates[13], landmark_px_coordinates[15])

                annocolor = (0,255,0)

            else:
                right_arm_angle = f"Right Arm Angle: ?"
                left_arm_angle = f"Left Arm Angle: ?"
                annocolor = (0, 0, 255)
                right_arm_annotation_position = (20, 200)
                left_arm_annotation_position = (20, 225)
            cv2.putText(frame, 
                        f"{right_arm_angle}", 
                        right_arm_annotation_position, 
                        cv2.FONT_HERSHEY_SIMPLEX, 
                        0.5, # font size 
                        annocolor, #font color
                        1, #thickness
                        2) # line type
            cv2.putText(frame, 
                        f"{left_arm_angle}", 
                        left_arm_annotation_position, 
                        cv2.FONT_HERSHEY_SIMPLEX, 
                        0.5, # font size 
                        annocolor, #font color
                        1, #thickness
                        2) # line type


        bordered_frame = pad_borders(frame)

        cv2.imshow("Pose Analysis", bordered_frame)

        key = cv2.waitKey(frame_delay) & 0xFF


        # Break Loop

        if key == ord('q'):
            break
        elif key == 27:
            break
        elif key == 32:
            while True:
                key2 = cv2.waitKey(1) & 0xFF
                if key2 == 32:   # resume on space
                    break
                elif key2 == ord('q') or key2 == 27:  # allow exit during pause
                    video.release()
                    cv2.destroyAllWindows()
                    sys.exit()
        elif key == ord('k'):
            compute_frames = (compute_frames + 1) % 2
        elif key == ord('o'):
            visualize_angles = (visualize_angles + 1) % 2
            

    video.release()
    cv2.destroyAllWindows()
    if loss_counter>0:
        print(f"Success Rate = {round(success_counter / (success_counter + loss_counter))}")
    else:
        print("no loss")
vfra('videoplayback.mp4', 2)




"""
0: nose

11: left shoulder, 12: right shoulder

13: left elbow, 14: right elbow

15: left wrist, 16: right wrist

23: left hip, 24: right hip

25: left knee, 26: right knee

27: left ankle, 28: right ankle

"""
